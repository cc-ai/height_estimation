#!/usr/bin/env python
# coding: utf-8


from math import tan, atan, sin, cos, pi, radians, degrees
import json
import os
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm


def get_metric_depth(depth, far = 1000):
    """ Get depth array in meters. We follow the method explained in the documentation of the CARLA dataset :
        https://carla.readthedocs.io/en/stable/cameras_and_sensors/
    
    Arguments:
        depth {np.array} -- depth map array (from RGB(A) image extracted with CARLA simulator), this is linear depth 
        far {float} -- how far the depth sensor sees in meters
    
    Returns:
        depth_metric {np.array}
         -- depth map in meters
    """
    depth_metric = (depth[:,:,0] + depth[:,:,1]*256 + depth[:,:,2]*(256*256)) / (256*256*256 - 1)
    depth_metric = depth_metric * far
    return (depth_metric)


def get_intrinsic_matrix(H,W,FOV = 100):
    """ Get matrix of camera intrinsics
    
    Arguments:
        H {int} -- height of the images, in pixels 
        W {int} -- width of the images, in pixels 
        FOVx {float} -- horizontal field of view in degrees
        
    Returns: 
        Kc {np.array} -- 3*3 matrix of camera parameters
       
    """
    FOVx = radians(FOV)
    print(FOVx)
    print(degrees( tan(radians(FOV)/2)* (H/W) ))
    FOVy = 2*atan((H/W) * tan(radians(FOV)/2))
    print(FOVy)
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


def get_3D_coords(depth_metric, inv_Kc, camera_height = 1.4, epsilon = -15):
    """ Get 3D coordinates from metric depth map 
    
    Arguments:
        depth_metric {np.array} -- metric depth map array
        FOVx {float} -- horizontal field of view in degrees
        epsilon {float} -- camera pitch in degrees
        
    Returns: 
        Kc {np.array} -- 3*3 matrix of camera parameters
       
    """
    H, W = depth_metric.shape
    #get grid of pixel coordinates (2D)
    coord1, coord2 =  np.mgrid[range(H), range(W)]
    coords_plane = np.c_[coord2.ravel(), coord1.ravel()]
    #add third dimension: depth, put minus because in the pinhole camera model the image is upside down
    coords3 = np.append(coords_plane, -np.expand_dims(depth_metric.flatten(), axis = 1), axis = 1)    

    coords3[:, 0] = np.multiply(coords3[:, 0],coords3[:, 2])
    coords3[:, 1] = np.multiply(coords3[:, 1],coords3[:, 2])

    #get the coordinates in the camera coordinate system
    coords_proj = np.transpose(np.matmul(inv_Kc , np.transpose(coords3)))
    
    #take into account rotation due to camera pitch (suppose roll is 0 )
    rotation = np.array(
        [[1, 0, 0], [0, cos(epsilon), -sin(epsilon)], [0, sin(epsilon), cos(epsilon)]]
    )
    # we actually right apply the transpose of the rotation matrix  which is the same as left applying the transpose matrix 
    #from camera coordinate system to real
    coords_proj_ =np.matmul(coords_proj, np.transpose(rotation)) 
    
    #pinhole inverses left right too 
    coords_proj_[:,0] = -coords_proj_[:,0]
    #add translation (camera level is 0 but in real it's camera_height above ground)
    coords_proj_[:,1] += camera_height
    return(coords_proj_)

##############################  VISUALIZATIONS  #####################################################################

def make_gif(img_path, path_root, coords_proj_, max_threshold = 1.4, step = 0.05, duration = 3, zero_ratios_len = 3):
    """Store a gif of an image, with zero_ratios_len images without flood before flooding the image
    one step at a time, until max_height

    Args:
        img_path (Path): where to find the image
        path_root (str): folder were to save the gifs
        coords_proj_ (np.array): 3D coordinates (metric units)
        max_threshold (float): flood until max_threshold (meters)
        step (float): how fine the grid search is
        duration (int): final gif length in seconds
        zero_ratios_len (int): how many frames to prepend to the gif,  without floods

    Returns:
        None: saves gif to disk
    """
    img_path = Path(img_path)
    images = []
    img = np.array(Image.open(img_path).convert('RGB'))
    H, W, _ = img.shape
    threshold = 0.
    while threshold < max_threshold:
        # flood everything under threshold
        flood = np.where(coords_proj_[:,1]<=threshold)
        colors = np.array(img.copy().reshape(H*W, 3))
        #color the pixels in light blue
        colors[flood, :] = [114, 139, 161]
        images.append(Image.fromarray(colors.reshape(H, W, 3)))
        threshold += step

    images = [Image.fromarray(img)] * zero_ratios_len + images
    gif_name = "{}_{}_{}.gif".format(str(img_path.stem), str(max_threshold).replace('.', '-'), str(step).replace('.', '-'))
    print(gif_name)
    
    images[0].save(
        os.path.join(path_root, gif_name),
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=1000 * duration / len(images),
        loop=1)
    
def plot_logdepth(depth_metric, save_path = None):
    """ Get grayscale image of metric depth map. We actually plot log depth for better visualization
    
    Arguments:
        depth_metric {np.array} -- metric depth map
        save_path {str} -- full path of where to save the log depth image 
        
    Returns: 
        logimg {PIL Image} -- image of log depth
        -- saves the image to save_path if specified
    """
    logdepth = np.log(depth_metric)
    logimg = Image.fromarray((logdepth - np.min(logdepth))/(np.max(logdepth) - np.min(logdepth))*255).convert('L')
    if not save_path is None:
        logimg.save(save_path)
    return(logimg)



######################## GENERATE HEIGHT DATASET FROM CARLA DEPTH IMAGES  ###########################################

def generate_height_dataset(depth_paths, save_path, camera_params, rgb_paths = None, freq = 100): 
    """ Generate dataset of numpy arrays of depth. 
        Sky (and all pixels that are further way from the camera than the camera far parameter) are imputed to np.inf
        Saves gifs if paths to original images are specified
    
    Arguments:
        depth_paths {list} -- list of file paths to the depth maps images from the CARLA simulator
        save_path {str} -- path of folder to save the numpy arrays of height 
        camera_params {str} -- path of json/txt file with camera parameters 
        rgb_paths {list} -- list of file paths to the images from the CARLA simulator, 
                            needs to be in the same order as the depth paths, useful only for the visualization
                            Used to make gifs of flood
        freq {int} -- save a gif every freq image in the folder
        
    Returns: 
        None -- saves .npy arrays of heights, sky (too far pixels) is imputed to np.inf
    """
    
    #read camera parameters
    with open(camera_params) as json_file:
        cam_params = json.load(json_file)
    far = cam_params['far']
    FOV = cam_params['FOVx']
    camera_height = float(cam_params['camera_height'])
    epsilon = radians(cam_params['epsilon'])
    img_height = cam_params['img_height']
    img_width = cam_params['img_width']
    
    Kc = get_intrinsic_matrix(img_height,img_width,FOV)   
    inv_Kc = np.linalg.inv(Kc)
    
    viz_flag = False
    if not rgb_paths is None:
        viz_flag = True 
        
    for index, depth_path in tqdm(enumerate(depth_paths)):
        depth_path =  Path(depth_path)
        depth = np.array(Image.open(depth_path))
        
        depth_metric = get_metric_depth(depth, far)
        sky0, sky1 = np.where(depth_metric==far)
        H, W = depth_metric.shape
        coords_proj_ = get_3D_coords(depth_metric, inv_Kc, camera_height, epsilon)

        save = os.path.join(save_path, str(depth_path.stem) + '_height.npy')
        
        height_array = coords_proj_[:, 1].reshape((H,W))
        
        #shift heigths to start at 0 
        height_array -= np.min(height_array)
        #clip sky to -1
        height_array[sky0, sky1] = -1
        #save height array
        np.save(save, height_array)
        
        if viz_flag and index%freq == 0:
            make_gif(rgb_paths[index], save_path, coords_proj_, max_threshold = camera_height, step = 0.1, duration = 3, zero_ratios_len = 3)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-depth_folder', help='path to folder with depth*.png files')
    parser.add_argument('-save_path', help='path to folder where to save depth arrays')
    parser.add_argument('-cam_params', help='path to json/txt file of camera parameters')
    parser.add_argument('-rgb_folder', help='path to folder with rgb*.png files', default=None, required = False)
    parser.add_argument('-freq', help='save gifs every freq image', default=100, required = False)
    args = parser.parse_args()
    print(args)
    print("Generating depth dataset")
    p = Path(args.save_path)
    print(os.curdir)
    p.mkdir(parents=True, exist_ok=True)
    print("Saving to " + args.save_path)
    depth_paths = sorted(list(Path(args.depth_folder).glob('depth*.png')))
    if not args.rgb_folder is None : 
        rgb_paths = sorted(list(Path(args.rgb_folder).glob('rgb*.png')))
    else:
        rgb_paths = None
    generate_height_dataset(depth_paths, args.save_path, args.cam_params, rgb_paths, int(args.freq))
    print("Done")