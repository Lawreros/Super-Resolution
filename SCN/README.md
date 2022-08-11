# Robust Single Image Super-Resolution via Deep Networks With Sparse Prior

## File list:
- image_generation.ipynb = notebook for generating training and testing image patches from both pngs and nii files
- MRI_SCN.ipynb = notebook containing both the BSD100 trained model and the MRI-trained model, also includes the training and testing functions
- test_images_.zip = zipped MRI images for testing the MRI-trained model
- MRI_run_SGD_v100_57.p = file containing weights for MRI-trained model, loaded in the MRI_SCN.ipynb when instantiating the MRI-trained model
- SCN_BSD_final.p = file containing weights for BSD100-trained model, loaded in the MRI_SCN.ipynb when instantiating the BSD100-trained model
