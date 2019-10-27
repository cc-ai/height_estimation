
## Directory Structure

This directory contains the source code for the height estimation pipeline. 

`BoundingBox3D`: a PyTorch implementation of the paper [3D BoundingBox Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496) (Mousavian et al.), from [this repo](https://github.com/skhadem/3D-BoundingBox). We used the pre-trained weights for the 3D BoundingBox net and YOLOv3 weights provided there.  
The metric dimensions of the objects are regressed around the mean values of the dimensions of the objects encountered in the KITTI dataset on which the object detection model was trained.  
For now, we only consider a subset of the detected objects, corresponding to the classes cat, truck and pedestrian. 
The reason behind this choice is that in our pipeline to get the height of objects in images, we are crossing this information with semantic segmentation information that is obtained with a model trained on a dataset with CityScapes labels (see `segmentation`). The direct intersection between KITTI and CityScapes labels is reduced to the classes `car` and `truck`. But in a later iteration of the model, we would like to find a relevant matching between classes, e.g. between classes  `pedestrian` and `person sitting` in KITTI and `person` in Cityscapes. Below is a table with the average dimensions (in meters) of the objects encountered in the training dataset. 

| Object | Counts in KITTI dataset | Height | Width | Length |
| ------------- | ------------- |------------- |------------- |------------- |
| car | 28742 |1.52608343 |  1.62858987 | 3.88395449  | 
| truck  | 1094  | 3.25170932 | 2.58509141 |10.10907678  | 

The 3D Bounding Box Model will regress the dimensions of detected objects around these values. 


`data_utils` : code to fetch image data with the GSV API

`Megadepth`: implementation of [MegaDepth: Learning Single-View Depth Prediction from Internet Photos](http://www.cs.cornell.edu/projects/megadepth/) (Li et al. ) from [this repo](https://github.com/zhengqili/MegaDepth). 
`test.py`contains functions for the inference on a single image. We also wrote a version to augment data and average on the inference. The default augmentations are the following : original image, vertical flip, vertical flip + brightness and contrast variation, and two brightness and constrast variations. 
The idea behind this was to try and smooth the ouput depth map from Megadepth by averaging the depths obtained on the images withe these various transformations. 


`segmentation`: semantic segmentation model from DeepLab with Cityscapes labels. 

We also tried the pre-trained  segmentation model from  NVIDIA paper [Improving Semantic Segmentation via Video Propagation and Label Relaxation](https://github.com/NVIDIA/semantic-segmentation). The model is trained on KITTI dataset and also uses Cityscapes labels. 

`core_utils.py`: functions to recover 3D coordinates (relative, non metric) using camera parameters and depthmap inputs with the pinhole camera model, and then match these coordinates to metric values using reference objects (assumed to be on the ground) in the image. 

`viz.py`: functions to plot and save figures of each step of the pipeline (2D bounding box, segmentation, image with 'flood' mask)

`demo_notebook.ipynb`: notebook to test the pipeline and get a mask of all the points that are below a certain height specified in meters. 
`demo_short.ipynb`: if you just want to see the ouput images without having to go through all intermediate steps. 
