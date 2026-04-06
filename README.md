# Homework 6: Advanced Image Classification

## Learning Goals:

This assignment is designed to function as a follow up to the basic image classification assignment. Our learning goals include:

1. Use Tensorflow in Python to implement computer vision methods
2. Train Convolutional Neural Network (CNN) to classify images 
3. Evaluate the results of a CNN classifier and interpret

## Background 

This week's assignment for DA 351 is based on an article titled "Building a Bird Recognition App and Large Scale Dataset With Citizen Scientists: The Fine Print in Fine-Grained Dataset Collection", which was authored by the SE(3) Computer Vision Group at Cornell University and presented at the Computer Vision and Pattern Recognition (CVPR) Conference in Boston in 2015. They worked with "citizen scientists and domain experts" to develop a "high quality dataset containing 48,562 images of North American birds with 555 categories, part annotations and bounding boxes" (see https://vision.cornell.edu/se3/building-a-bird-recognition-app-and-large-scale-dataset-with-citizen-scientists-the-fine-print-in-fine-grained-dataset-collection/). 

The dataset they created, `NABirds`, is really fantastic, but it's also really huge (9.8 GB compressed on my disk and 9.99 GB after you unzip it)! For our version, we will work the Caltech-UCSD Birds-200-2011 Dataset (CUB 200), which was the go-to bird recognition/classification dataset prior to `NABirds`. CUB 200 is still fairly large at 1.2 GB compressed, but that's a lot less than 9.8 GB. 

Either dataset will exceed Github's file size limits, so you will need to download the data yourself. Here is a link to the dataset:

| Dataset | Download Link | Filename | 
|---|---|---|
| CUB 200 | https://data.caltech.edu/records/65de6-vp158 | CUB_200_2011.tgz |

__Note:__ If you want to try the `NABirds` dataset, you're more than welcome. Accessing it requires you to provide your email and agree to the terms and conditions. 

| Dataset | Download Link | Filename | 
|---|---|---|
| NABirds | https://dl.allaboutbirds.org/nabirds | nabirds.tar.gz |


