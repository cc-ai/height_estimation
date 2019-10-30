#!/usr/bin/env python
# coding: utf-8

# # Generate binary masks of all points below a certain metric height

import os
import time
from PIL import Image
from math import sin, cos, pi, tan
import numpy as np
import cv2
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import vgg
from BoundingBox3D.yolo.yolo import cv_Yolo
from BoundingBox3D.torch_lib.Dataset import *
from BoundingBox3D.torch_lib import Model, ClassAverages
from BoundingBox3D.library.Math import *
from BoundingBox3D.library.Plotting import *
import os
from data_utils import download_SV_img as download_sv
from MegaDepth import test as megadepth_test
from MegaDepth.options.train_options import TrainOptions
from viz import show_2D_bbox, show_flood, show_seg_mask, generate_binary_mask
from segmentation import segmentation
from core_utils import filter_segmentation, get_thresh_coords_megadepth
from math import atan, tan, radians
import yaml
import sys



def generate_masks(config_file = './config/config_default.yaml'):

    with open(config_file, 'r') as stream:
        config = yaml.safe_load(stream)

    # Define some gps location for street view 
    if not config['addresses_from_csv']:
        addresses =  config['addresses']
    key = config['key']
    save_SV_path =  config['save_SV_path']
    save_DM_path = config['save_DM_path']
    crop = config['crop']
    megadepth_path = config['megadepth_path']
    weights_path = config['weights_path']
    yolo_path = config['yolo_path']
    classes = config['classes']
    img_size = config['img_size']
    H, W = img_size
    FOVx = 2*atan(tan(radians(config['init_FOVx']))*((W/2)-crop)/(W/2))
    FOVy = 2*atan(tan(radians(config['init_FOVy']))*((H/2)-crop)/(H/2))
    thresholds = config['thresholds']
    segmentation_path = config['segmentation_path']
    output_path = config['output_path']
    save_seg_path = config['save_seg_path']

    if not os.path.exists(output_path):
        os.mkdir(output_path)


    # Download images from the adresses
    output_paths = download_sv.get_images([download_sv.param_block(address, key) for address in addresses],save_SV_path, crop_print=crop) 

    depth_paths = []
    for img_file in output_paths:
        path_depth = megadepth_test.get_depthmap_img(img_file, save_DM_path, img_size, megadepth_path)
        depth_paths.append(path_depth + '.npy')

    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        # TODO: load bins from file or something
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    yolo = cv_Yolo(yolo_path)
    averages = ClassAverages.ClassAverages()
    angle_bins = generate_bins(2)

    paths_detect = []
    for ind, img_file in enumerate(output_paths):
        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)
        detections = yolo.detect(yolo_img)
        
        masks = segmentation.segment_19classes([img_file], segmentation_path, save_seg_path)
        show_seg_mask(masks[0])
        mask = np.load(masks[0])
        if len(detections) == 0:
            #if no object is detected, output mask corresponding to the road
            im_mask = Image.fromarray(mask == 0 ).convert('1')
            im_mask.save(os.path.basename(img_file[:-4]) + '_'+ 'ground_mask.jpg')
        else : 
            detections_keep = []
            for elem in detections: 
                print(elem.detected_class)
                if elem.detected_class  in classes.keys():
                    detections_keep.append(elem)
            print("keeping" + str(len(detections_keep)) + '/' + str(len(detections)) + " detected objects")
            detections = detections_keep
            show_2D_bbox(img_file, detections, save_path =  None)
            #paths_detect.append(path2D_bbox)

            depth = np.load(depth_paths[ind])

            #Enter metric threshold
            for threshold in thresholds:
                coords, threshs = get_thresh_coords_megadepth(img, threshold, depth, FOVx, FOVy, detections, mask, classes, model,pix_threshold = 0.3, epsilon = 0)
                show_flood(img, np.mean(threshs), coords, save_path = save_seg_path + os.path.basename(img_file[:-4]) + '_' + str(threshold).replace('.', '-') +'_flood.jpg')
                generate_binary_mask(img, np.mean(threshs), coords, output_path + os.path.basename(img_file[:-4]) + '_'+ str(threshold).replace('.', '-') + '_mask.jpg')
                print( output_path + os.path.basename(img_file[:-4]) + '_'+ str(threshold).replace('.', '-') + '_mask.jpg')

        print("Done")


if __name__=='__main__':
    generate_masks(sys.argv[1])