# End-to-End height estimator

Real images of floods do not come with annotation on the height of the flood. Height information is very important for our purpose of generating realistiv visualizations of the impacts of climate change. Ideally, we would like to control the level of the flood  in order to match climate predictions, or at least physically plausible levels. In addition, we would like to be able to specify this level in an easily understandable unit (meters ideally) to link this more easily with the reality.  

We propose to leverage data from our simulator and train a height estimation model from single-view images. Indeed, images of houses flooded to any chosen height can be generated in the simulator.

While methods for single image depth estimation have been investigated for many years, we have found no work in height estimation from street view scenes images. However, these two problems have some similarities, so we took inspiration from depths estimators to build our height estimator. 

### Ground truth height maps

Ground truth depths maps are extracted directly from our simulator and  ground truth height maps can be computed using the pinhole camera model ([See the derivation](https://github.com/cc-ai/height_estimation/tree/master/simulated_world)). 
Since the depth maps are provided in metric units, the heights obtained from the height maps are also metric. 
Height is relative to some origin, and one initial idea to fix the origin would be to take the lowest point as the zero, and assume that it corresponds to the ground.
However, we noticed that there were sometimes some errors in the height maps (most probably due to encoding + precision that decreases the further away the objects are): some pixels far (in depth) from the camera would end up with heights that seem like outlier to the distribution (e.g. very negative values far from the rest of the height distribution) and these errors are not so straightforward to fix.  
The aforementioned idea of picking the lowest point as zero would lead to very inconsistent flood areas, if we then pick a physically plausible flood level in a range of (0 - 10m) for example. 

Nevertheless, we still need to have a consistent way of picking the zero. We chose to take the middle front pixel (bottom row of the image) as the reference for the zero, and shift the rest of the dataset accordingly. 

### The network

Following the success of  [MegaDepth]( http://www.cs.cornell.edu/projects/megadepth/ ) on the task of single-view depth estimation, we chose to train an [hourglass network]( https://arxiv.org/pdf/1604.03901.pdf ) to predict metric height.

The code for the network was inspired from [this implementation]( https://github.com/yifjiang/relative-depth-using-pytorch ) of the paper [Single-Image Depth Perception in the Wild](https://arxiv.org/pdf/1604.03901.pdf).

 ![hourglass](https://github.com/cc-ai/height_estimation/blob/master/hourglass/docs/hourglass.JPG) 

Architecture of the hourglass network (Image modified from [here]( https://arxiv.org/pdf/1604.03901.pdf )). 
Each block is a modified inception module, except blocks H which are Conv 3x3.


