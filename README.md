# Super-Resolution Paper Replication
This repository is ment to contain all of my experimentation with and replication of the different Super-Resolution (SR) papers and concepts that I come across. Each paper will be given its own folder, with a README.md for each which further explains its contents.

**Everyone is free to use the code found within this repository as they see fit (depending on the licensing of the codebases/publications I have provided links for), I only ask that you cite the relevant sources for each of these SR implementations (and myself if applicable)**

The `gen_utils` directory contains the classes and functions that I have created that perform functions that are used across the different SR implementations.

## Convolutional Neural Networks with Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
Contained in the `CNNIL` directory, with the offical github repository found here: https://github.com/lilygeorgescu/3d-super-res-cnn

Summary: 


```
@misc{Georgescu-2020,
    title={Convolutional Neural Networks with Intermediate Loss for 3D Super-Resolution of CT and MRI Scans},
    author={Mariana-Iuliana Georgescu and Radu Tudor Ionescu and Nicolae Verga},
    year={2020}, 
    book={IEEE Access}
}
```


## Robust Single Image Super-Resolution via Deep Networks With Sparse Prior
Contained in the `SCN` directory,

```
@ARTICLE{7466062,
  author={Liu, Ding and Wang, Zhaowen and Wen, Bihan and Yang, Jianchao and Han, Wei and Huang, Thomas S.},
  journal={IEEE Transactions on Image Processing}, 
  title={Robust Single Image Super-Resolution via Deep Networks With Sparse Prior}, 
  year={2016},
  volume={25},
  number={7},
  pages={3194-3207},
  doi={10.1109/TIP.2016.2564643}}
```


## File list:
- image_generation.ipynb = notebook for generating training and testing image patches from both pngs and nii files
- MRI_SCN.ipynb = notebook containing both the BSD100 trained model and the MRI-trained model, also includes the training and testing functions
- test_images_.zip = zipped MRI images for testing the MRI-trained model
- MRI_run_SGD_v100_57.p = file containing weights for MRI-trained model, loaded in the MRI_SCN.ipynb when instantiating the MRI-trained model
- SCN_BSD_final.p = file containing weights for BSD100-trained model, loaded in the MRI_SCN.ipynb when instantiating the BSD100-trained model
