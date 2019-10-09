#!/usr/bin/env python
# coding: utf-8

from matplotlib import patches
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

def show_2D_bbox(img_file, detections, save_path = None):
    # plot 2D bounding box
    im = cv2.imread(img_file)[...,::-1]#np.array(Image.open(img_file), dtype=np.uint8)
    print(im.shape)
    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(im)
    for detection in detections:
        # Create a Rectangle patch
        rect = patches.Rectangle(detection.box_2d[0],detection.box_2d[1][0] - detection.box_2d[0][0], detection.box_2d[1][1] - detection.box_2d[0][1],linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)
    if not save_path is None: 
        plt.savefig(save_path)
    plt.show()
    
def show_seg_mask(mask_file):
    """
    mask_file : numpy array
    """
    mask = np.load(mask_file)
    plt.imshow(mask, cmap = 'tab20')

def show_flood(img, threshold, coords, save_path = None):
    fig = plt.figure()
    red = [255,0,0]
    #flood everything under threshold
    flood = np.where(coords[:,:,1]< threshold)
    colors = img.copy().reshape(512,512, 3)
    colors[flood[0], flood[1], : ] = red
    flooded_im = colors.reshape(512,512,3)

    img_pil = cv2.cvtColor(flooded_im, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(img_pil)
    if not save_path is None: 
        im.save(save_path)
    return(im)

