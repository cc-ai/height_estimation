# End-to-End height estimator

Real images of floods do not come with annotation on the height of the flood. Height information is very important for our purpose of generating realistic visualizations of the impacts of climate change. Ideally, we would like to control the level of the flood  in order to match climate predictions, or at least physically plausible levels. In addition, we would like to be able to specify this level in an easily understandable unit (meters ideally) to link this more easily with the reality.  

We propose to leverage data from our simulator and train a height estimation model from single-view images. Indeed, images of houses flooded to any chosen height can be generated in the simulator.

While methods for single image depth estimation have been investigated for many years, we have found no work in height estimation from street view scenes images. However, these two problems have some similarities, so we took inspiration from depths estimators to build our height estimator. 

### Quick usage
#### Training the model
Check the `config_train.yaml` file in the `config` folder to know which parameters to specify in your config file. 
All images must be in the same folder, and all height maps as well. You need to specify the path to text files containing the basenames of the images and the height maps for the train and test set. 
You can train the model running : 

`python scripts/train.py --config config/config_train.yaml`

#### Inference
Check the `config_test_default.yaml` file in the `config` folder to know which parameters to specify in your config file. 

`python scripts/test.py --config config/config_test_default.yaml --input INPUT_IMG_DIR --output_folder OUTPUT_dir --checkpoint PATH_TO_CHECKPOINT`

### Target
Our current model outputs metric height maps from street view images inputs. 
Currently, we predict the height of all the pixels that do not correspond to sky. 
see **Next steps** for ideas on how to handle sky for further work. 

### Ground truth height maps

Ground truth depths maps are extracted directly from our simulator and  ground truth height maps can be computed using the pinhole camera model ([See the derivation](https://github.com/cc-ai/height_estimation/tree/master/simulated_world)). 
Since the depth maps are provided in metric units, the heights obtained from the height maps are also metric. 
Height is relative to some origin, and one initial idea to fix the origin would be to take the lowest point as the zero, and assume that it corresponds to the ground.
However, we noticed that there were sometimes some errors in the height maps (most probably due to encoding + precision that decreases the further away the objects are): some pixels far (in depth) from the camera would end up with heights that seem like outlier to the distribution (e.g. very negative values far from the rest of the height distribution) and these errors are not so straightforward to fix.  
The aforementioned idea of picking the lowest point as zero would lead to very inconsistent flood areas, if we then pick a physically plausible flood level in a range of (0 - 10m) for example. 

Nevertheless, we still need to have a consistent way of picking the zero. We chose to take the middle front pixel (bottom row of the image) as the reference for the zero, and shift the rest of the dataset accordingly. 

The sky is assigned to *inf*.

### The network

Following the success of  [MegaDepth]( http://www.cs.cornell.edu/projects/megadepth/ ) on the task of single-view depth estimation, we chose to train an [hourglass network]( https://arxiv.org/pdf/1604.03901.pdf ) to predict metric height.

The code for the network was inspired from [this implementation]( https://github.com/yifjiang/relative-depth-using-pytorch ) of the paper [Single-Image Depth Perception in the Wild](https://arxiv.org/pdf/1604.03901.pdf).

 ![hourglass](https://github.com/cc-ai/height_estimation/blob/master/hourglass/docs/hourglass.JPG) 

Architecture of the hourglass network (Image modified from [here]( https://arxiv.org/pdf/1604.03901.pdf )). 
Each block is a modified inception module, except blocks H which are Conv 3x3.

### Loss functions 

As of now, we are using pixel wise L2 loss with a mask for the pixels corresponding to the sky (because the sky is *inf* in our ground truth).
We could imagine not masking the sky is the target is chosen differently (See **Next Steps**). 

In our case, we care about getting heights that are close to the ground right, acare a little less about getting the heights of the parts that are > 10m high right, since it will not affect our masks of areas that should be flooded. 
We implemented a loss that considers this aspect of the problem, but haven't studied extensively which parameters to choose.  

### A note on training data

We used this model separately on two datasets from simulators. 
We trained this model on a first set of 600 images from our simulator. 
We also trained this model on 8500 images extracted from the CARLA simulator under different weather conditions (6500 training, 2000 testing). However the scenes in this dataset can be very similar, and were taken at a high frequency leading to some frames being redundant. More over, when acquiring these images, no special care was given to make sure that they could correspond to street scenes encountered in real images. For example, we spotted frames where the back of a pedestrian close to the camera would fill the whole field-of-view. When using this dataset, some care would have to be taken to remove "problematic" images.

### Next steps

From our early study results, we notice that the height maps predicted are not smooth. For example, iscontinuities in texture/color on a building facade lead to discontinuities in heights. We could imagine getting a **smoother output** using the following :  

- Adding [**scale-invariant gradient loss**](https://arxiv.org/pdf/1612.02401.pdf) which has been used successfully for [3D Ken Burns effect from single view image](https://arxiv.org/abs/1909.05483). 
This loss would penalize relative height errors between neighbouring pixels. 
In the task of depth estimation *"This loss stimulates the network to compare depth values within a local neighbourhood for each pixel. It emphasizes depth discontinuities, stimulates sharp edges in the depth map and increases smoothness within homogeneous regions."* 

- Post-processing the height map (e.g. apply MRF) 

**Handling sky**: It is not possible to predict the sky area in our target if it is kept at *inf*. It is not that much of an issue if we assume that we compute segmentation masks for the sky (at inference time too to "fill" the sky area) but with a well chosen target we could make the prediction on the whole image. 
One idea would be to take the target as the inverse depth <img src =https://latex.codecogs.com/gif.latex?$1/(depth + 1)$> . 
The sky would then be mapped to 0. 
However we would need to make some modifications in our ground truth. Indeed, our current way of choosing the zero level does not guarantee that there are no negative heights. 

**Using real data**  
