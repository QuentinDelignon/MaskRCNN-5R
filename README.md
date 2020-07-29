# MaskRCNN-5R
This is an implementation of MaskRCNN Pretrained on COCO and fine tuned on a custom dataset to segment images containing toys with masks. This was part of a bin picking project at Arts & Métiers Paris. 
## Introduction
I uploaded this project to GitHub in order to help anybody who wish to finetune a Mask RCNN model. The user may use an Annotator like VGG Image Annotator to make his own dataset or use another one he downloaded. This repo is made to be an example and a stepping stone to achieve this goal.
## The Network 
The network is the same as presented in the paper.
## The Training Method 
We get the pretrained network from Pytorch then we train the network with our small dataset.
## How To
Clone the repo. Change the folder path of the dataset in training.py and also the saves folder. Execute training.py from its parent folder.
# Results 
We will not detail further the results. We obtain more than 90% accuracy and the sytem generates good masks for our challenge. The user can go further if he wants.  
<img src="https://github.com/QuentinDelignon/MaskRCNN-5R/blob/media/result_1.png" width="300" >
<img src="https://github.com/QuentinDelignon/MaskRCNN-5R/blob/media/result_14.png" width="300" >
