#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from math import radians, atan, tan, sin, cos

def convert_depth(im_array, far):
    """
    convert RGB depth image as np.array to array containing metric depth values.
    The depth is encoded in the following way: 
    - The information from the simulator is (1 - LinearDepth (in [0,1])). 
        far corresponds to the furthest distance to the camera included in the depth map. 
        LinearDepth * far gives the real metric distance to the camera. 
    - depth is first divided in 31 slices encoded in R channel with values ranging from 0 to 247
    - each slice is divided again in 31 slices, whose value is encoded in G channel
    - each of the G slices is divided into 256 slices, encoded in B channel
    In total, we have a discretization of depth into N = 31*31*256 - 1 possible values, covering a range of 
    far/N meters.   
    Note that, what we encode here is 1 - LinearDepth so that the furthest point is [0,0,0] (that is sky) 
    and the closest point[255,255,255] 
    The metric distance associated to a pixel whose depth is (R,G,B) is : 
        d = (far/N) * [((255 - R)//8)*256*31 + ((255 - G)//8)*256 + (255 - B)]
                
    """
    R = im_array[:,:,0]
    G = im_array[:,:,1]
    B = im_array[:,:,2]
    
    R = ((247 - R) //8).astype(float)
    G = ((247 - G)//8).astype(float)
    B = (255 - B).astype(float)
    depth = ((R*256*31 + G*256 + B).astype(float))/ (256*31*31 - 1)
    return(depth*far)

def get_camera_params_from_json(json_file, water_height = 0.45):
    """
    We only have access to the absolute height of the camera, not the height wrt ground.
    Note that ground is not flat in the simulated world. 
    In our dataset, we collected images such that the water level is always ~0.45 m above ground,
    the ground reference being where the camera sees. 
    As an approximation, we will consider this is where the camera sits. 
    
    FOV is the vertical field of view, in degrees
    """
    with open(json_file) as file:
        data = json.load(file)
    FOVy = data['CameraFOV']
    height = data['CameraPosition']['y'] - data['WaterLevel'] + water_height
    if data['CameraRotation']['x'] <= 180:
        pitch = -data['CameraRotation']['x']
    else : 
        pitch = 360 - data['CameraRotation']['x']
    far = data['CameraFar']
    return {'cam_height': height, 'pitch': pitch, 'FOVy' : FOVy, 'far' : far}
    
def get_intrinsic_matrix_vFOV(H,W,FOVy):
    """ Get matrix of camera intrinsics using vertical field of view
    
    Arguments:
        H {int} -- height of the images, in pixels 
        W {int} -- width of the images, in pixels 
        FOVy {float} -- vertical field of view in degrees
        
    Returns: 
        Kc {np.array} -- 3*3 matrix of camera parameters
       
    """
    FOVy = radians(FOVy)
    FOVx = 2*atan((W/H) * tan(FOVy/2))
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

def rotation_matrix_pitch(pitch):
    """
    Get rotation matrix around x-axis 
    
    Arguments:
        pitch {float} -- pitch in degrees
        
    Returns: 
       {np.array} -- 3*3 rotation matric
    """
    pitch_ = radians(pitch)
    return (np.array(
                        [[1, 0, 0], 
                         [0, cos(pitch_), -sin(pitch_)],
                         [0, sin(pitch_), cos(pitch_)]]
                     )
              )

def get_3D_coords(depth_metric, inv_Kc, camera_height, pitch = 0):
    """ Get 3D coordinates from metric depth map 
    
    Arguments:
        depth_metric {np.array} -- metric depth map array
        inv_Kc {float} -- horizontal field of view in degrees
        pitch {float} -- camera pitch in degrees
        
    Returns: 
        Kc {np.array} -- 3*3 matrix of camera parameters
       
    """
    H, W = depth_metric.shape
    #get grid of pixel coordinates (2D)
    coord1, coord2 =  np.mgrid[range(H), range(W)]
    
    #create array (H,W,3) from which to compute 3d coords (x (horizontal), y (vertical), z (depth)) for each pixel
    coords3 = np.zeros((H,W,3))
    coords3[:,:,-1] =  depth_metric

    coords3[:,:,0] = np.multiply(coord2, coords3[:,:,2])
    coords3[:,:, 1] = np.multiply(coord1, coords3[:,:,2])
    
    #take into account rotation due to camera pitch (suppose roll is 0 )
    rotation = rotation_matrix_pitch(pitch)
    
    #get the coordinates in the camera coordinate system
    # transpose of rotation matric is its inverse
    #  Conventionally, a positive rotation angle corresponds to a counterclockwise rotation. 
    coords_proj = np.transpose(np.dot(np.dot(np.transpose(rotation),inv_Kc),np.transpose(coords3.reshape(H*W,3))))
    
    #the pinhole model make a 180° rotation of the image
    #pinhole inverses left right 
    coords_proj[:,0] = -coords_proj[:,0]
    #pinhole inverses top bottom 
    #add translation (camera level is 0 but in real it's camera_height above ground)
    coords_proj[:,1] = - coords_proj[:,1]  + camera_height
    return(coords_proj)

def get_height(img_file, json_file,sky_value = np.inf,  water_height = 0.45):
    """
    Compute 3D coordinates and height map from simulated data
    Arguments:
        img_file{str} : path to depth img file
        json_file{str} : path to json
        water_height {float} : height of water wrt to ground in cm
    Returns: 
        depth_metric {np.array} : metric depth map 
        coords {np.array} : 3D coordinates array of size H*W*3, for each pixel, the coordinates are specified as (x (horizontal), y (vertical (height)), z (depth))
        height_array {np.array} : height map (2nd axis of the 3D coordinates, with the sky clipped at sky_value)
        params {dict} : dictionary of camera parameters
    """
    params = get_camera_params_from_json(json_file, water_height)
    print(params)
    FOVy = params['FOVy']
    cam_height =  params['cam_height']
    pitch = params['pitch']
    far = params['far']
    depth = np.array(Image.open(img_file))
    
    H,W, _ = depth.shape
    
    depth_metric = convert_depth(depth, far)
    
    sky0, sky1 = np.where(depth_metric==far)
    
    Kc = get_intrinsic_matrix_vFOV(H,W,FOVy)    
    inv_Kc = np.linalg.inv(Kc)
    
    coords = get_3D_coords(depth_metric, inv_Kc, cam_height, pitch)
    height_array = coords[:, 1].reshape((H,W))
    
    #shift heigths to start at 0 
    #clip sky to some value 
    height_array[sky0, sky1] = sky_value

    return(depth_metric, coords, height_array, params)

def set_zero_reference(height_array, leave_untouched = None):
    """
    Get shifted height array where zero is taken as corresponding to the bottom row middle pixel's value
    Typically, leave untouched sky pixels. 
    Arguments:
        img_file{str} : path to depth img file
        json_file{str} : path to json
        water_height {float} : height of water wrt to ground in cm
    Returns: 
        depth_metric {np.array} : metric depth map 
    """
    H, W = height_array.shape 
    ref = height_array[H-1, W//2]
    if leave_untouched is not None:
        height_array[height_array != leave_untouched] -= ref
    else:
        height_array -= ref
    return(height_array)

def fix_ground_outlier(height_array, seg, depth_array, depth_threshold):
    """
    Alternative way of setting the zero reference - /!\ drawback is that it is not a consistent way of choosing the referece
    It can happen that there are outlier height points - especially for far away points
    The goal is to set the zero to correspond to some point of the ground in order to compute height
    However, due to outlier points, we canot just shift the whole height map using the minimum height value. 
    Empirically we notice that most of the "problems" occur on far away points, so we take the min value to make the zero level shift
    to be the min value of height on points that belong to ground and that are less than a certain depth threshold away from the camera.
    
    - look at pixels segmented as water or terrain(later on we would like to merge terrain and ground but for the moment we only 
    have segmentation maps of flooded images.
    - look at points that a less than depth_threshold away from the camera 
    - take the min value
    - make the shift with this value 
    
    """
    #[0,0,255, 255] corresponds to water segmentation, [255, 97, 0, 255] to terrain]
    indices =  np.where((np.all(seg == [255, 97, 0, 255], axis = -1) | np.all(seg == [0, 0, 255, 255], axis = -1))& (depth_array< depth_threshold) )  
    if len(indices[0]) ==0:
        return(height_array)
    else: 
        min_value = np.min(height_array[indices[0], indices[1]])
        #reset 0 
        print(min_value)
        height_array = height_array - min_value
        return(height_array)
    
############### Visualizations #############################################################################################
def normalize(array):
    return((array - np.min(array))/(np.max(array) - np.min(array)))

def plot_logdepth(depth, far): 
    """
    Visualize depth map - plot normalized log depth for nicer visuals
    """
    im = Image.fromarray(256*(normalize(np.log(depth/far)))).convert('L')
    fig = plt.imshow(im)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)



