#!/usr/bin/env python
# coding: utf-8

from math import tan, atan, sin, cos, radians, degrees
import numpy as np
from BoundingBox3D.torch_lib.Dataset import DetectedObject
import torch
from BoundingBox3D.torch_lib import Model, ClassAverages
    
def get_rotation_matrix(epsilon):
    """ Get rotation matrix around x-axis
    Arguments:
        epsilon {float} -- rotation angle (in radians)
    Returns: 
        {np.array} -- 3*3 rotation matrix
    
    """
    return (np.array([[1,0,0], [0, cos(epsilon), -sin(epsilon)], [0, sin(epsilon), cos(epsilon)]]))


def get_intrinsic_matrix(H,W,FOV = 120):
    """ Get matrix of camera intrinsics
    
    Arguments:
        H {int} -- height of the images, in pixels 
        W {int} -- width of the images, in pixels 
        FOVx {float} -- horizontal field of view in degrees
        
    Returns: 
        Kc {np.array} -- 3*3 matrix of camera parameters
       
    """
    FOVx = radians(FOV)
    FOVy = 2*atan(H/W * tan(radians(FOV)/2))
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

    
def get_3D_coords(depth, FOV, epsilon = radians(10)):
    """
    Recover 3D coordinates (not necessarily metric)
    Arguments:
        depth {np.array} -- depth values (not necessarily metric units) of each pixel
        FOV {float} -- horizontal field of view in degrees
        epsilon {float} -- camera pitch in radians
        
    Returns: 
        coords_proj_ {np.array} -- 3D coordinates
       
    """
    #The initial FOV in the API was 120°. But we center-cropped the image, taking out margins of 20 pixels on all sides (to get rid of the watermarks)
    # We then resized the image to 512*512 so that we could apply MegaDepth on it ( some pixels are dilated)
    
    H, W = depth.shape

    Kc = get_intrinsic_matrix(H,W,FOV)   
    inv_Kc = np.linalg.inv(Kc)

    
    coord1, coord2 =  np.mgrid[range(H), range(W)]
    coords_plane = np.c_[coord2.ravel(), coord1.ravel()]
    #add third dimension: depth, put minus because in the pinhole camera model the image is upside down
    coords3 = np.append(coords_plane, -np.expand_dims(depth.flatten(), axis = 1), axis = 1)    
    coords3[:, 0] = np.multiply(coords3[:, 0],coords3[:, 2])
    coords3[:, 1] = np.multiply(coords3[:, 1],coords3[:, 2])

    #get the coordinates in the camera coordinate system
    #this is the rotation matrix from the non rotated to rotated (camera) coordinate system
    coords_proj = np.transpose(inv_Kc @ np.transpose(coords3))
    
    #take into account rotation due to camera pitch (suppose roll is 0 )
    rotation = get_rotation_matrix(epsilon)
    # we actually right apply the transpose of the rotation matrix  which is the same as left applying the transpose matrix 
    #from camera coordinate system to real
    coords_proj_ =coords_proj@np.transpose(rotation)
    
    #pinhole inverses left right too 
    coords_proj_[:,0] = -coords_proj_[:,0]
    
    return(coords_proj_.reshape((H,W, 3)))
    

def get_depthmap_megadepth(depth):
    """
    depth : depthmap array obtained with MegaDepth/ MegaDepth oututs 1/depth so we recover initial depth output
    """
    depth_ = (1/(depth/255))
    depth_ /= np.max(depth_)
    depth_ *= 255
    return(depth_)
    
def filter_segmentation(detections,mask,classes, pix_threshold = 0.3):
    """
    classes : dictionary of classes (works for one to one mapping)
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
    The Image coordinate system is H,W and the 3D reconstructed system is W,H,Depth
    
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
    #print(dim)
    height_object = dim[0]
    print("height of the detected object of class " + detection.detected_class + " in meters is : " + str(height_object))
    return(hmin+ threshold * (hmax-hmin) /height_object)

def get_thresh_coords_megadepth(img, threshold, depth, FOV, detections, mask , classes, model, pix_threshold = 0.3,  epsilon = radians(10)):
    depth_ = get_depthmap_megadepth(depth)
    H,W = depth.shape
    cam = get_intrinsic_matrix(H,W,FOV)
    
    coords = get_3D_coords(depth_, FOV, epsilon)
    keep = filter_segmentation(detections,mask, classes , pix_threshold)
    thresholds = []
    for index in keep : 
        single_thresh = get_coord_threshold_single_detection(threshold, detections[index], coords, mask, img, cam, model, classes)
        thresholds.append(single_thresh)
    return(coords, thresholds)
    
    
#######################WORK IN PROGRESS ################################################################################ 
def map_func_KITTI_Cityscapes(label_detection, mask):
    """
    hardcoded version that outputs binary mask (array) of a class, making the labels correspond to the classes in the following way : 
    classes:  {'car': 13, 'truck': [14 or 15 ], 'person_sitting': 10, 'pedestrian': 10, 'cyclist': [12 and 11], }
    map labels from KITTI dataset to Cityscapes classes
    """
    if label_detection == 'car':
        return (mask == 13)
    if label_detection == 'truck':
        return ((mask == 14) | (mask == 15))
    if label_detection == 'pedestrian' or label_detection == 'person_sitting' or label_detection == 'person':
        return (mask == 11)
    if label_detection == 'rider' or label_detection == 'bicycle':
        return (((mask == 18) | (mask == 12)) & (mask==18).sum()>0  & (mask==12).sum()>0)
#def filter_segmentation_(detections,mask, map_func, pix_threshold = 0.3):
#    """
#    classes : dictionary of classes (works for one to one mapping)
#    only keep detection if detected class is present in the segmentation is takes at least pix_threshold % of the bbox
#    
#    """
#    #check if we can use the segmentation 
#    keep = []
    
#    for index, detection in enumerate(detections):
#        #class_ = classes[detection.detected_class]
#        xmin, xmax, ymin, ymax = (max(0,detection.box_2d[0][0]), detection.box_2d[1][0], max(0,detection.box_2d[0][1]), detection.box_2d[1][1])
#        print(detection.detected_class)
#        surface_box = (xmax - xmin)*(ymax - ymin)

#        pix_detected = np.sum(map_func(detection.detected_class, mask[ymin:ymax, xmin:xmax]))
#        if not pix_detected is None : 
#            if pix_detected/surface_box > pix_threshold:
#keep.append(index)
#    print(keep)
#    return (keep)
