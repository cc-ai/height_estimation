# height_estimation
Estimate objects' heights in Street View images

One of the challenges in generating realistic images of floods is to control the level of the flood, in order to match climate predictions. We propose to introduce height information in our model in order to generate and respect the geometry of the scene. 

Here we present the two types of approaches that we have tried so far to estimate height in street scene images : "geometrical approaches" which consist in recovering the 3D metric geometry of the scene, and  "end-to-end approaches" which consist in predicting height from an image input. 

You can refer to [this page](https://cc-ai.github.io/MUNIT/index.html) to understand how height estimation fits in our goal of generating images of the impacts of climate change. 

### Description

`pinhole` contains a notebook to show how to recover 3D coordinates from an image with a depth map and camera parameters using the pinhole camera model.  

`src` contains the pipeline for height estimation in an image using geometrical approaches. 

 - Clone the content of this [repo]((https://github.com/skhadem/3D-BoundingBox) into the 3DBoundingBox folder.  
    `git clone https://github.com/skhadem/3D-BoundingBox BoundingBox3D`
- Download the weights of the 3D bounding box estimator from [that same repo](https://github.com/skhadem/3D-BoundingBox). To do so, go into the `BoundingBox3D/weights`folder and run `get_weights.sh`.

- Set up `src`as a package run:  `python setup.py develop`

Have a look at the `demo_notebook.ipynb` to see how all the pieces come together. 

`hourglass` is a first model for end-to-end height estimtion
 
`simulated_world` contains code to generate the height maps dataset from the images and depth maps from our simulator. 

