# Super-Resolution Paper Replication
This repository is meant to contain all of my experimentation with (and replication of) the different Super-Resolution (SR) papers and concepts that I come across. Each paper will be given its own folder, with a README.md for each which further explains its contents.

**Everyone is free to use the code found within this repository as they see fit (depending on the licensing of the codebases/publications I have provided links for), I only ask that you cite the relevant sources for each of these SR implementations (and myself if applicable)**

The `gen_utils` directory contains the classes and functions that I have created that perform functions that are used across the different SR implementations. These are mainly related to the loading/processing/sampling of input `.nii`, `.png`, and `.dcm` files.

## Convolutional Neural Networks with Intermediate Loss for 3D Super-Resolution of CT and MRI Scans
Contained in the `CNNIL` directory. The citation for the paper is found below, with the offical github repository found here: https://github.com/lilygeorgescu/3d-super-res-cnn

### Summary: 
This paper explores using two seperate 2D SR CNN models to accomplish 3D SR. Effectively, if you have a 3D image with axes [x,y,z] and you want to double its size, you first have a model double the height and width of slices along a given axis:

```mermaid
  flowchart LR
    3D img : [x,y,z] --> [x,y,0] & [x,y,1] & ... & [x,y,z] --> CNN_1 --> [2x,2y,0] & [2x,2y,1] & ... & [2x,2y,z] --> 3D img : [2x,2y,z]
```



```
dimensions of input 3D image = [x,y,z]

input -> z 2D images of size [x,y] -> model_1 -> z 2D images of size [2x, 2y]

combine z 2D images to get output
dimensions of output 3D image = [2x,2y,z]

```
 Then you use a second model which only doubles the remaining dimension of the 3D image by taking 2D slices from a different orientation ()

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