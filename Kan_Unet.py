import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import math
from pytorch_wavelets import DWTForward
from logging import config
from mamba_ssm.modules.mamba_simple import Mamba
from .OutlookAttention import OutlookAttention
from .lowlight_model.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d



class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, device='cpu'):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad, device=self.device)

        return out


class EqualConv2d(nn.Module):
    def __init__(
            self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ConvLayer(nn.Sequential):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            downsample=False,
            blur_kernel=[1, 3, 3, 1],
            bias=True,
            activate=True,
            device='cpu'
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), device=device))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel, device=device))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], device='cpu'):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, device=device)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


#################################################################################

# ##################################################  KAN  #############################################################
class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
                .expand(in_features, -1)
                .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
            self,
            layers_hidden,
            grid_size=5,  # 原来的代码
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )
            

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            # x = layer(x)  # 原来的代码
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )


class PVMLayer(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)  # 归一化层
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v2"
        )

        # self.proj = nn.Linear(input_dim, output_dim) 
        self.proj = nn.Linear(input_dim, output_dim)

        self.skip_scale = nn.Parameter(torch.ones(1))
       

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)  
        # print(x.shape)
        # exit()
       
        B, C = x.shape[:2]
        assert C == self.input_dim  
        n_tokens = x.shape[2:].numel() 
        img_dims = x.shape[2:]  
        
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)  
        x_norm = self.norm(x_flat)  

       
        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2) 
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2) 

        x_mamba = self.norm(x_mamba)  
        
        B1, N1, C1 = x_mamba.shape
        x_mamba = x_mamba.view(-1, C1)
        
        x_mamba = self.proj(x_mamba) 
        x_mamba = x_mamba.view(B1, N1, -1)
       

        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        
        return out



class Channel_Att_Bridge(nn.Module):
   
    def __init__(self, c_list, split_att='fc', mlp_ratio=4.):
        super().__init__()
        c_list_sum = sum(c_list) - c_list[-1]
        
        self.split_att = split_att 
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.get_all_att = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        
        
        self.att1 = KAN(layers_hidden=[c_list_sum, int(c_list_sum * mlp_ratio), c_list[0]]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[0], 1)
        self.att2 = KAN(layers_hidden=[c_list_sum, int(c_list_sum * mlp_ratio), c_list[1]]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[1], 1)
        self.att3 = KAN(layers_hidden=[c_list_sum, int(c_list_sum * mlp_ratio), c_list[2]]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[2], 1)
        self.att4 = KAN(layers_hidden=[c_list_sum, int(c_list_sum * mlp_ratio), c_list[3]]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[3], 1)
        self.att5 = KAN(layers_hidden=[c_list_sum, int(c_list_sum * mlp_ratio), c_list[4]]) if split_att == 'fc' else nn.Conv1d(c_list_sum, c_list[4], 1)


        self.sigmoid = nn.Sigmoid() 


    def forward(self, t1, t2, t3, t4, t5):
        att = torch.cat((self.avgpool(t1),
                         self.avgpool(t2),
                         self.avgpool(t3),
                         self.avgpool(t4),
                         self.avgpool(t5)), dim=1)  
        att = self.get_all_att(att.squeeze(-1).transpose(-1, -2))
        if self.split_att != 'fc':
            att = att.transpose(-1, -2)


        b1, n1, c1 = att.shape
        att = att.view(-1, c1)
        

        att1 = self.sigmoid(self.att1(att).view(b1, n1, -1))
        att2 = self.sigmoid(self.att2(att).view(b1, n1, -1))
        att3 = self.sigmoid(self.att3(att).view(b1, n1, -1))
        att4 = self.sigmoid(self.att4(att).view(b1, n1, -1))
        att5 = self.sigmoid(self.att5(att).view(b1, n1, -1))

        if self.split_att == 'fc':
            att1 = att1.transpose(-1, -2).unsqueeze(-1).expand_as(t1)
            att2 = att2.transpose(-1, -2).unsqueeze(-1).expand_as(t2)
            att3 = att3.transpose(-1, -2).unsqueeze(-1).expand_as(t3)
            att4 = att4.transpose(-1, -2).unsqueeze(-1).expand_as(t4)
            att5 = att5.transpose(-1, -2).unsqueeze(-1).expand_as(t5)
        else:
            att1 = att1.unsqueeze(-1).expand_as(t1)
            att2 = att2.unsqueeze(-1).expand_as(t2)
            att3 = att3.unsqueeze(-1).expand_as(t3)
            att4 = att4.unsqueeze(-1).expand_as(t4)
            att5 = att5.unsqueeze(-1).expand_as(t5)

        return att1, att2, att3, att4, att5


class Spatial_Att_Bridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_conv2d = nn.Sequential(nn.Conv2d(2, 1, 7, stride=1, padding=9, dilation=3),
                                           nn.Sigmoid())

    def forward(self, t1, t2, t3, t4, t5):
        t_list = [t1, t2, t3, t4, t5]
        att_list = []
        for t in t_list:
            avg_out = torch.mean(t, dim=1, keepdim=True)
            max_out, _ = torch.max(t, dim=1, keepdim=True)
            att = torch.cat([avg_out, max_out], dim=1)
            att = self.shared_conv2d(att)
            att_list.append(att)
       
        return att_list[0], att_list[1], att_list[2], att_list[3], att_list[4]


class SC_Att_Bridge(nn.Module):
    def __init__(self, c_list, split_att='fc'):
        super().__init__()

        self.catt = Channel_Att_Bridge(c_list, split_att=split_att)
        self.satt = Spatial_Att_Bridge()

    def forward(self, t1, t2, t3, t4, t5):
        r1, r2, r3, r4, r5 = t1, t2, t3, t4, t5

        satt1, satt2, satt3, satt4, satt5 = self.satt(t1, t2, t3, t4, t5)
       
        t1, t2, t3, t4, t5 = satt1 * t1, satt2 * t2, satt3 * t3, satt4 * t4, satt5 * t5

        r1_, r2_, r3_, r4_, r5_ = t1, t2, t3, t4, t5
        t1, t2, t3, t4, t5 = t1 + r1, t2 + r2, t3 + r3, t4 + r4, t5 + r5 
        catt1, catt2, catt3, catt4, catt5 = self.catt(t1, t2, t3, t4, t5)
        t1, t2, t3, t4, t5 = catt1 * t1, catt2 * t2, catt3 * t3, catt4 * t4, catt5 * t5

        return t1 + r1_, t2 + r2_, t3 + r3_, t4 + r4_, t5 + r5_


class Kan_UNet(nn.Module):

    def __init__(self, num_classes=4, input_channels=1, c_list=[8, 16, 24, 32, 48, 64],
                 split_att='fc', bridge=True, mlp_ratio=8.):
        
        super().__init__()

        self.bridge = bridge

    
        self.encoder1 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )
       

        self.encoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[0], output_dim=c_list[1])
        )

        self.encoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[1], output_dim=c_list[2])
        )


        self.encoder4 = nn.Sequential(
            PVMLayer(input_dim=c_list[2], output_dim=c_list[3])
        )
        self.encoder5 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[4])
        )
        self.encoder6 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[5])
        )
        
        self.short_map1 = OutlookAttention(input_channels)  # B, H, W, C  

        self.encoder21 = nn.Sequential(
            nn.Conv2d(input_channels, c_list[0], 3, stride=1, padding=1),
        )

        self.short_map2 = OutlookAttention(c_list[0])  # B, H, W, C  

        self.encoder22 = nn.Sequential(
            nn.Conv2d(c_list[0], c_list[1], 3, stride=1, padding=1),
        )

        self.short_map3 = OutlookAttention(c_list[1])  # B, H, W, C  
        self.encoder23 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[2], 3, stride=1, padding=1),
        )

        self.short_map4 = OutlookAttention(c_list[2])  # B, H, W, C  
        self.encoder24 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[3], 3, stride=1, padding=1),
        )

        self.short_map5 = OutlookAttention(c_list[3])  # B, H, W, C  
        self.encoder25 = nn.Sequential(
            nn.Conv2d(c_list[3], c_list[4], 3, stride=1, padding=1),
        )

        self.short_map6 = OutlookAttention(c_list[4])  # B, H, W, C  
        self.encoder26 = nn.Sequential(
            nn.Conv2d(c_list[4], c_list[5], 3, stride=1, padding=1),
        )

        self.fusion0 = ConvLayer(c_list[0] * 2, c_list[0], 1, device='gpu')
        self.fusion1 = ConvLayer(c_list[1] * 2, c_list[1], 1, device='gpu')
        self.fusion2 = ConvLayer(c_list[2] * 2, c_list[2], 1, device='gpu')
        self.fusion3 = ConvLayer(c_list[3] * 2, c_list[3], 1, device='gpu')
        self.fusion4 = ConvLayer(c_list[4] * 2, c_list[4], 1, device='gpu')
        self.fusion5 = ConvLayer(c_list[5] * 2, c_list[5], 1, device='gpu')
        self.fusion6 = ConvLayer(num_classes * 2, num_classes, 1, device='gpu')

        
        if bridge:
            self.scab = SC_Att_Bridge(c_list, split_att)
            print('SC_Att_Bridge was used')
       

        self.decoder1 = nn.Sequential(
            PVMLayer(input_dim=c_list[5], output_dim=c_list[4])
        )
        self.decoder2 = nn.Sequential(
            PVMLayer(input_dim=c_list[4], output_dim=c_list[3])
        )
        self.decoder3 = nn.Sequential(
            PVMLayer(input_dim=c_list[3], output_dim=c_list[2])
        )
        
        self.decoder4 = nn.Sequential(
            nn.Conv2d(c_list[2], c_list[1], 3, stride=1, padding=1),
        )
        self.decoder5 = nn.Sequential(
            nn.Conv2d(c_list[1], c_list[0], 3, stride=1, padding=1),
        )
        
        self.ebn1 = nn.GroupNorm(4, c_list[0])  # 8
        self.ebn2 = nn.GroupNorm(4, c_list[1])  # 16
        self.ebn3 = nn.GroupNorm(4, c_list[2])  # 24
        self.ebn4 = nn.GroupNorm(4, c_list[3])  # 32
        self.ebn5 = nn.GroupNorm(4, c_list[4])  # 48
        self.dbn1 = nn.GroupNorm(4, c_list[4])  # 48
        self.dbn2 = nn.GroupNorm(4, c_list[3])  # 32
        self.dbn3 = nn.GroupNorm(4, c_list[2])  # 24
        self.dbn4 = nn.GroupNorm(4, c_list[1])  # 16
        self.dbn5 = nn.GroupNorm(4, c_list[0])  # 8
       

        self.final = nn.Conv2d(c_list[0], num_classes, kernel_size=1)  
        
        self.apply(self._init_weights)  
   
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
       
        elif isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        
        x1 = x.permute(0, 2, 3, 1)  # B,H,W,C

        short_map1 = self.ebn1(self.encoder21(self.short_map1(x1).permute(0, 3, 1, 2)))  # B, c0, H, W

        long_map1 = self.ebn1(self.encoder1(x))  # B, c0, H, W

        out = F.gelu(F.max_pool2d(self.fusion0(torch.cat([short_map1, long_map1], dim=1)), 2, 2))  # b, c0, H/2, W/2
        
        t1 = out  # b, c0, H/2, W/2

        x2 = out.permute(0, 2, 3, 1)  # B, H/2, W/2, c0
        short_map2 = self.ebn2(self.encoder22(self.short_map2(x2).permute(0, 3, 1, 2)))  # B, c1, H/2, W/2
        long_map2 = self.ebn2(self.encoder2(out))  # B, c1, H/2, W/2
        out = F.gelu(F.max_pool2d(self.fusion1(torch.cat([short_map2, long_map2], dim=1)), 2, 2))  # B, c1, H/4, W/4

        t2 = out  # b, c1, H/4, W/4

        
        x3 = out.permute(0, 2, 3, 1)  # B, H/4, W/4, c1
        short_map3 = self.ebn3(self.encoder23(self.short_map3(x3).permute(0, 3, 1, 2)))  # B, c2, H/4, W/4
        long_map3 = self.ebn3(self.encoder3(out))  # B, c2, H/4, W/4
        out = F.gelu(F.max_pool2d(self.fusion2(torch.cat([short_map3, long_map3], dim=1)), 2, 2))  # B, c2, H/8, W/8
        t3 = out  # b, c2, H/8, W/8

        x4 = out.permute(0, 2, 3, 1)  # B, H/8, W/8, c2
        short_map4 = self.ebn4(self.encoder24(self.short_map4(x4).permute(0, 3, 1, 2)))  # B, c3, H/8, W/8
        long_map4 = self.ebn4(self.encoder4(out))  # B, c3, H/8, W/8
        out = F.gelu(F.max_pool2d(self.fusion3(torch.cat([short_map4, long_map4], dim=1)), 2, 2))  # B, c3, H/16, W/16
    
        t4 = out  # b, c3, H/16, W/16

        
        out31 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out)), scale_factor=(2, 2), mode='bilinear',
                                     align_corners=True))  # b, c2, H/8, W/8 卷积解码
        out31 = torch.add(out31, t3)  # b, c3, H/8, W/8
        out21 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out31)), scale_factor=(2, 2), mode='bilinear',
                                     align_corners=True))  # b, c1, H/4, W/4  卷积解码
        out21 = torch.add(out21, t2)  # b, c2, H/4, W/4
        out11 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out21)), scale_factor=(2, 2), mode='bilinear',
                                     align_corners=True))  # b, c0, H/2, W/2  卷积解码
        out11 = torch.add(out11, t1)  # b, c0, H/2, W/2
        out01 = F.interpolate(self.final(out11), scale_factor=(2, 2), mode='bilinear',
                              align_corners=True)  # b, num_class, H, W

       
        x5 = out.permute(0, 2, 3, 1)  # b, H/16, W/16, c3
        short_map5 = self.ebn5(self.encoder25(self.short_map5(x5).permute(0, 3, 1, 2)))  # B, c4, H/16, W/16
        long_map5 = self.ebn5(self.encoder5(out))  # B, c4, H/16, W/16
        out = F.gelu(F.max_pool2d(self.fusion4(torch.cat([short_map5, long_map5], dim=1)), 2, 2))  # B, c4, H/32, W/32
    
        t5 = out  # b, c4, H/32, W/32

        
        if self.bridge:
            t1, t2, t3, t4, t5 = self.scab(t1, t2, t3, t4, t5)
        

        x6 = out.permute(0, 2, 3, 1)  # B, H/32, W/32, c4
        short_map6 = self.encoder26(self.short_map6(x6).permute(0, 3, 1, 2))  # b, c5, H/32, W/32
        long_map6 = self.encoder6(out)  # b, c5, H/32, W/32
        out = F.gelu(self.fusion5(torch.cat([short_map6, long_map6], dim=1)))  # b, c5, H/32, W/32
        
        out5 = F.gelu(self.dbn1(self.decoder1(out)))  # b, c4, H/32, W/32
        out5 = torch.add(out5, t5)  # b, c4, H/32, W/32


        out4 = F.gelu(F.interpolate(self.dbn2(self.decoder2(out5)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c3, H/16, W/16
        out4 = torch.add(out4, t4)  # b, c3, H/16, W/16

        out3 = F.gelu(F.interpolate(self.dbn3(self.decoder3(out4)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c2, H/8, W/8
        out3 = torch.add(out3, t3)  # b, c2, H/8, W/8

        out2 = F.gelu(F.interpolate(self.dbn4(self.decoder4(out3)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c1, H/4, W/4
        out2 = torch.add(out2, t2)  # b, c1, H/4, W/4

        out1 = F.gelu(F.interpolate(self.dbn5(self.decoder5(out2)), scale_factor=(2, 2), mode='bilinear',
                                    align_corners=True))  # b, c0, H/2, W/2
        out1 = torch.add(out1, t1)  # b, c0, H/2, W/2

        out0 = F.interpolate(self.final(out1), scale_factor=(2, 2), mode='bilinear',
                             align_corners=True)  # b, num_class, H, W
       
       
        return out0, out01


if __name__ == '__main__':
    block = Kan_UNet(num_classes=1, input_channels=1).cuda()  
    input = torch.rand(3, 1, 64, 64).to('cuda')
    output = block(input)
    # print(input.size(), output.size())
    print(input.shape, output.shape)
