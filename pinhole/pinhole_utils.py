#!/usr/bin/env python
# coding: utf-8

import numpy as np
from math import sin, cos, tan, atan, degrees, radians



def get_3D_coords(depth_metric, inv_Kc,  pitch = 0):
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
    coords_proj[:,1] = - coords_proj[:,1] 
    return(coords_proj)


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
    Args:
        pitch (float): pitch of the camera in degrees
    Returns:
        np.array : rotation matrix 
    """
    pitch_ = radians(pitch)
    return (np.array(
                        [[1, 0, 0], 
                         [0, cos(pitch_), -sin(pitch_)],
                         [0, sin(pitch_), cos(pitch_)]]
                     )
              )

### Choose threshold for mask generation
def get_coord_threshold(threshold, coords_proj_):
    """Scale a [0; 1] threshold to the scale of coords_proj_

    Args:
        threshold (float): [0;1] float ratio
        coords_proj_ (np.array): non-metric projected coordinates

    Returns:
        float: scaled threshold
    """
    return coords_proj_.min() + threshold * (coords_proj_.max() - coords_proj_.min())

