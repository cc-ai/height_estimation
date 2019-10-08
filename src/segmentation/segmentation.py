# Pytorch implementation of deeplab with resnet_34_8s_cityscapes_best

import numpy
import torch
import torch.nn as nn
from segmentation.resnet import resnet34
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from torch.utils.data import DataLoader, Dataset
import os
import cv2
from PIL import Image
import tqdm 
import argparse
from torchvision import transforms

class Resnet34_8s(nn.Module):

    def __init__(self, num_classes=1000):
        super(Resnet34_8s, self).__init__()

        # Load the pretrained weights, remove avg pool
        # layer and get the output stride of 8
        resnet34_8s = resnet34(fully_conv=True,
                                      pretrained=True,
                                      output_stride=8,
                                      remove_avg_pool_layer=True)

        # Randomly initialize the 1x1 Conv scoring layer
        resnet34_8s.fc = nn.Conv2d(resnet34_8s.inplanes, num_classes, 1)

        self.resnet34_8s = resnet34_8s

        self._normal_initialization(self.resnet34_8s.fc)

    def _normal_initialization(self, layer):
        layer.weight.data.normal_(0, 0.01)
        layer.bias.data.zero_()

    def forward(self, x, feature_alignment=False):
        input_spatial_dim = x.size()[2:]

        if feature_alignment:
            x = adjust_input_image_size_for_proper_feature_alignment(x, output_stride=8)

        x = self.resnet34_8s(x)

        x = nn.functional.upsample_bilinear(input=x, size=input_spatial_dim)

        return x




class MyDataset(Dataset):
    def __init__(self,root,transform=None):
        self.images=[root+f for f in os.listdir(root)]
        self.transform=transform
        self.root = root

    def __getitem__(self,index):
        path=self.images[index]
        image=Image.open(path).convert('RGB')
        if self.transform is not None:
           image=self.transform(image)
        return image
    def __len__(self):
        return len(self.images)

    def getPaths(self):
        return os.listdir(self.root)

class MyDataset_from_list(Dataset):
    def __init__(self,list_img,transform=None):
        self.images=list_img
        self.transform=transform

    def __getitem__(self,index):
        path=self.images[index]
        image=Image.open(path).convert('RGB')
        if self.transform is not None:
           image=self.transform(image)
        return image
    def __len__(self):
        return len(self.images)

    def getPaths(self):
        return self.images
    
def segment_19classes(path_list, weight_pth, dir_mask, batch_size = 1,  size_mask = None):
    """
    path_list : list of paths of the images to segment
    Return : list of paths to saved masks (as numpy arrays)
    """
    if size_mask is None:
        size_mask = Image.open(path_list[0]).size
    # Perform image transformation
    valid_transform = transforms.Compose(
                [
                     transforms.Resize(size_mask),
                     transforms.ToTensor(),
                     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

    model = Resnet34_8s(num_classes=19).to('cuda')
    PATH  = weight_pth
    model.load_state_dict(torch.load(PATH))

    val_ds=MyDataset_from_list(path_list,transform = valid_transform)

    val_dl=torch.utils.data.DataLoader(val_ds,batch_size=batch_size,shuffle=False)

    model.eval()

    it_= 0 
    list_paths = val_ds.getPaths()
    masks_paths = []
    with torch.no_grad():
        for i_batch, images in enumerate(val_dl):
            imgs =images.to('cuda')
            out_batch= model(imgs.float()).cpu()
            res=out_batch.squeeze(0).max(0)
            np.save(os.path.join(dir_mask, os.path.basename(list_paths[i_batch][:-4]) +'.npy'), res[1]) 
    masks_paths.append(os.path.join(dir_mask, os.path.basename(list_paths[i_batch][:-4]) +'.npy' ))
    print("Segmentation done")
    return (masks_paths)