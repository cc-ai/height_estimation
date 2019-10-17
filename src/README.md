
## Directory Structure

This directory contains the source code for the height estimation pipeline. 

`BoundingBox3D`: a PyTorch implementation of the paper [3D BoundingBox Estimation Using Deep Learning and Geometry](https://arxiv.org/abs/1612.00496) (Mousavian et al.), from [this repo](https://github.com/skhadem/3D-BoundingBox). We used the pre-trained weights for the 3D BoundingBox net and YOLOv3 weights provided there. 

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
