# MRI_Contrast_classifier

This repo aims at creating classications models to predict MRI contrast from 2D Mri images.
The goal of this project if to build a robust net that can take 2D 1 channel MRI image of any shape, any view point, any field of view and output it's contrast.

The first network trained is designed to discriminate T1w against T2w.


# Dataset

The original training set is extracted from the public dataset[Spine Generic](https://github.com/spine-generic/data-multi-subject).
This Dataset contains 3D images and require proper prepocessing to be used.

## Preprocessing

The data has been preprocesse to make the network as robust as possible

* The dataset is filtered to keep T1w and T2w images paths only.
* The dataset is splited between train patients and test patients (20% test)
* Two object from the class "2D_dataset" are created. They encapsulate the label and the image path.
* At each training epoch, the model sees each 3D image once. Each time the image is randomly :
    - flipped
    - rotated (in a 3° range)
    - Shifted (0.1 range)
    - reframed (in a 2D fashon, with minimum size (30 * 30))

# Network Structure

The network used is a ResNet18 from pytorch library modified to handle 1 channel images and to output a 2D vector.




