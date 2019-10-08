# height_estimation
Estimate objects' heights in Street View images

### Description

`pinhole` contains a notebook to show how to recover 3D coordinates from an image with a depth map and camera parameters using the pinhole camera model.  

`src` contains a first pipeline for height estimation in an image. 
To set up `src`as a package run:  `python setup.py develop`
You will need to download the weights of the 3D bounding box estimator from [here](https://github.com/skhadem/3D-BoundingBox). TO do so, go into the `weights`folder and run `get_weights.sh`.
Have a look at the `demo_notebook.ipynb` to see how all the pieces come together. 
