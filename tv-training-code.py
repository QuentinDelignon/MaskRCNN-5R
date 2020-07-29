# Sample code from the TorchVision 0.3 Object Detection Finetuning Tutorial
# http://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
import json
import os
import numpy as np
import torch
from PIL import Image,ImageDraw

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T


def seg2mask(w,h,seg):
    img = Image.new('L',(w,h),0)
    ImageDraw.Draw(img).polygon(seg,outline=1,fill=1)
    img = np.array(img)
    return img

def make_target(annotations,idx,w,h):
    target = {'boxes':[],
              'labels':[],
              'image_id':idx,
              'area':[],
              'iscrowd':[],
              'masks':[]}
    for i in range(len(annotations)):
        num_class = 0
        if int(annotations[i]['image_id']) > idx:
            break
        if int(annotations[i]['image_id']) == idx:
            num_class += 1
            target['boxes'].append(annotations[i]['bbox'])
            target['labels'].append(annotations[i]['category_id'])
            target['area'].append(annotations[i]['area'])
            target['iscrowd'].append(annotations[i]['iscrowd'])
            target['masks'].append(seg2mask(w,h,annotations[i]['segmentation']))
    target['boxes'] = torch.tensor(target['boxes'],dtype=torch.uint8)
    target['labels'] = torch.tensor(target['labels'],dtype=torch.int64)
    target['image_id'] = torch.tensor(target['image_id'],dtype=torch.uint8)
    target['area'] = torch.tensor(target['area'],dtype=torch.float64)
    target['iscrowd'] = torch.tensor(target['iscrowd'],dtype=torch.uint8)
    target['masks'] = torch.tensor(target['masks'],dtype=torch.uint8)
    return target

class PennFudanDataset(object):
    def __init__(self, transforms):
        self.root = 'C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\images\\'
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        """
        Decompresser le JSON
        """
        json_file = open('annotations.json','r')
        json_data = json_file.readline()
        json_file.close()
        json_data = json.loads(json_data)
        self.annotations = json_data['annotations']
        self.imgs = json_data['images']


    def __getitem__(self, idx):
        # load images ad masks
        img_name = self.imgs[idx]['file_name']
        img_path = self.root+img_name
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        target = make_target(self.annotations,idx,self.imgs[idx]['height'],self.imgs[idx]['width'])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 9
    # use our dataset and defined transformations
    dataset = PennFudanDataset(get_transform(train=True))
    dataset_test = PennFudanDataset(get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-10])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        print('/////////////// Epoch N°%d ///////////////////'%(epoch))
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")

if __name__ == "__main__":
    main()
