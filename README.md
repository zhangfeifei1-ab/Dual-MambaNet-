
If our work is accepted, we will make the code open source.



# Dual-MambaNet+: A Lightweight Segmentation Model for Brain MRI Images

## Overview

Dual-MambaNet+ is a cutting-edge, lightweight segmentation model designed for the diagnosis and treatment of brain diseases by accurately segmenting brain tissue from MRI images. This model is particularly adept at handling high-resolution MRI images without losing local feature information, which is a common challenge in medical image analysis.

## Requirements

Before you begin, ensure you have met the following requirements:
Pytorch and some basic python packages: Torchio, Numpy, Scikit-image, SimpleITK, Scipy, Medpy, nibabel, tqdm
## Installation

To install Dual-MambaNet+, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/Dual-MambaNet-.git
   cd Dual-MambaNet-
   ```

2. Install the required packages:

## Usage

To use Dual-MambaNet+ for brain tissue segmentation, follow these steps:

1. **Data Preparation**: Ensure your MRI data is preprocessed and formatted correctly for the model.
2. **Training**: Train the model using the training dataset.
   ```
  python train_fully_supervised_2D_VIM.py --root_path ../path/to/your/data --exp OASIS1 --model Dual-MambaNet+ --max_iterations 40000 --batch_size 24  --num_classes 4

   ```
4. **Test**: python test_2D_fully.py --root_path ../path/to/your/data --exp OASIS1/xxx --model xxx

## Experiments

We have conducted experiments on the public brain MRI datasets OASIS1 and MRBrainS13. The results demonstrate that Dual-MambaNet+ achieves outstanding segmentation accuracy while maintaining a minimal parameter count and computational complexity.

## Contributing

We welcome contributions to Dual-MambaNet+. If you find any issues or have suggestions for improvements, please create an issue or submit a pull request.

## License

Dual-MambaNet+ is released under the MIT License. See the `LICENSE` file for more details.

## Acknowledgements

We would like to express our gratitude to the creators of the OASIS1 and MRBrainS13 datasets for providing the valuable MRI data used in our experiments. Additionally, we are thankful to the authors of Mamba-UNet and UltraLight VM-UNet for their work, which served as an invaluable reference and inspiration for the development of our Dual-MambaNet+ model. Furthermore, we appreciate the Mamba framework authors for providing us with powerful tools and code that played a significant role in our development process.

We are grateful for the efforts and wisdom of all these contributors, whose work has laid a solid foundation for our research.


This README provides a comprehensive guide to using Dual-MambaNet+ for brain tissue segmentation. If you have any questions or need further assistance, please do not hesitate to reach out.
