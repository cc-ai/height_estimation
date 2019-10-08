#!/usr/bin/env python
# coding: utf-8

from math import tan, atan, sin, cos, radians, degrees
import numpy as np
from BoundingBox3D.torch_lib.Dataset import DetectedObject
import torch
from BoundingBox3D.torch_lib import Model, ClassAverages
    
def get_rotation_matrix(epsilon):
    return(np.array([[1,0,0], [0, cos(epsilon), -sin(epsilon)], [0, sin(epsilon), cos(epsilon)]]))

def get_homogeneous_rotation_matrix(eps):
    rotation = np.array([[1, 0, 0, 0], [0, cos(eps), -sin(eps), 0], [0, sin(eps), cos(eps), 0]])
    return(rotation)

def get_intrinsic_matrix(H,W,FOVx, FOVy):
    cx = (W/2)
    cy = (H/2)

    fx = cx / tan(FOVx/2)
    fy = cy/ tan(FOVy/2)
    
    #camera intrinsic matrix
    Kc = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])
    return(Kc)


def get_camera_matrix(H,W,FOVx, FOVy, epsilon):
    Kc = get_intrinsic_matrix(H,W,FOVx, FOVy)
    rotation = get_homogeneous_rotation_matrix(epsilon)
    return (np.dot(Kc, rotation))
    
    
def get_3D_coords(depth, FOVx, FOVy, epsilon = radians(10)):
    """
    depth : array of depth values of each pixel
    """

    #The initial FOV in the API was 120°. But we center-cropped the image, taking out margins of 20 pixels on all sides (to get rid of the watermarks)
    # We then resized the image to 512*512 so that we could apply MegaDepth on it ( some pixels are dilated)
    
    H, W = depth.shape

    Kc = get_intrinsic_matrix(H,W,FOVx, FOVy)
    
    inv_Kc = np.linalg.inv(Kc)


    coord1, coord2 =  np.meshgrid(range(H), range(W)) 
    coords_plane = np.stack((coord1.flatten(), coord2.flatten()), axis=-1)
    coords3 = np.append(coords_plane,  -np.expand_dims(depth.flatten(), axis = 1), axis = 1)
    # minus sign on the third direction because the pinhole model inverses 
    coords3[:, 0] = coords3[:, 0]*coords3[:, 2]
    coords3[:, 1] = coords3[:, 1]*coords3[:, 2]
    
    def inv_Kc_mult(a):
        return(inv_Kc.dot(a))
    #get the coordinates in the camera coordinate system
    coords_proj = np.apply_along_axis(inv_Kc_mult, 1, coords3)

    #apply rotation matrix
    #this is the rotation matrix from the non rotated to rotated (camera) coordinate system
    rotation = get_rotation_matrix(epsilon)
    #we actually right apply the transpose of the rotation matrix  which is the same as left applying the matrix from camera coordinate system to real
    coords_proj_ = np.dot(coords_proj, rotation)
    coords_proj_ = coords_proj_.reshape((H,W,3))
    return(coords_proj_)

def get_depthmap_megadepth(depth):
    """
    depth : depthmap array obtained with MegaDepth/ MegaDepth oututs 1/depth so we recover initial depth output
    """
    depth_ = (1/(depth/255))
    depth_ /= np.max(depth_)
    depth_ *= 255
    return(depth_)


def filter_segmentation(detections,mask , classes, pix_threshold = 0.3):
    """
    only keep detection if detected class is present in the segmentation is takes at least pix_threshold % of the bbox
    
    """
    #check if we can use the segmentation 
    keep = []
    
    for index, detection in enumerate(detections):
        class_ = classes[detection.detected_class]
        xmin, xmax, ymin, ymax = (max(0,detection.box_2d[0][0]), detection.box_2d[1][0], max(0,detection.box_2d[0][1]), detection.box_2d[1][1])
        surface_box = (xmax - xmin)*(ymax - ymin)
        if np.sum(mask[ymin:ymax, xmin:xmax] == class_)/surface_box > pix_threshold:
            keep.append(index)
    print(keep)
    return (keep)



def get_coord_threshold_single_detection(threshold, detection,coords, mask, img, cam, model,classes):
    """
    Get correspondence of metric input vertical threshold in the 3D coordinates recovery units using one detected object as reference.
    This will be used to be averaged over multiple detected objects
    """
    
    class_ = classes[detection.detected_class]
    
    xmin, xmax, ymin, ymax = (max(0,detection.box_2d[0][0]), detection.box_2d[1][0], max(0,detection.box_2d[0][1]), detection.box_2d[1][1])
    condition = np.where(mask[ymin:ymax, xmin:xmax] == class_)
    segmented = coords[ymin:ymax, xmin:xmax, :][condition[0], condition[1], :]
    hmax, hmin = np.max(segmented[:,1]), np.min(segmented[:,1])
    print(hmax, hmin)
    #Get dimension
    
    detection_box = [(xmin, ymin), (xmax, ymax)]
    detectedObject = DetectedObject(img, detection.detected_class, detection_box, cam)
    input_img = detectedObject.img
    input_tensor = torch.zeros([1,3,224,224]).cuda()
    input_tensor[0,:,:,:] = input_img

    [orient, conf, dim] = model(input_tensor)
    averages = ClassAverages.ClassAverages()
    dim = dim.cpu().data.numpy()[0, :]
    dim += averages.get_item(detection.detected_class)
    print(dim)
    height_object = dim[1]
    print("height of the detected object of class " + detection.detected_class + " in meters is : " + str(height_object))
    return(hmin+ threshold * (hmax-hmin) /height_object)

def get_thresh_coords_megadepth(img, threshold, depth, FOVx, FOVy, detections, mask, classes, model, pix_threshold = 0.3,  epsilon = radians(10)):
    depth_ = get_depthmap_megadepth(depth)
    H,W = depth.shape
    cam = get_camera_matrix(H,W,FOVx, FOVy, epsilon)
    
    coords = get_3D_coords(depth_, FOVx, FOVy, epsilon = radians(10))
    keep = filter_segmentation(detections,mask , classes, pix_threshold = 0.3)
    thresholds = []
    for index in keep : 
        single_thresh = get_coord_threshold_single_detection(threshold, detections[index], coords, mask, img, cam, model, classes)
        thresholds.append(single_thresh)
    return(coords, thresholds)
    