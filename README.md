# height_estimation
Estimate objects' heights in Street View images

### Description

`pinhole` contains a notebook to show how to recover 3D coordinates from an image with a depth map and camera parameters using the pinhole camera model.  

`src` contains a first pipeline for height estimation in an image. 

 - Clone the content of this [repo]((https://github.com/skhadem/3D-BoundingBox) into the 3DBoundingBox folder.  
    `git clone https://github.com/skhadem/3D-BoundingBox BoundingBox3D`
Download the weights of the 3D bounding box estimator from [that same repo](https://github.com/skhadem/3D-BoundingBox). To do so, go into the `BoundingBox3D/weights`folder and run `get_weights.sh`.

- Set up `src`as a package run:  `python setup.py develop`

Have a look at the `demo_notebook.ipynb` to see how all the pieces come together. 
