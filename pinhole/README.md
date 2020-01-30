### Pinhole camera model
 We consider a pinhole camera model. 
The mapping from 3D to 2D coordinates described by a pinhole camera is a perspective projection followed by a 180° rotation in the image plane.
Let <img src = " https://latex.codecogs.com/gif.latex?$K_c$"> the camera intrinsic matric.  
<img src = "https://latex.codecogs.com/gif.latex?$$\mathbf{K_c}&space;=&space;\left[\begin{array}&space;{rrr}f_x&space;&&space;0&space;&&space;c_x&space;\\&space;0&space;&&space;f_y&space;&&space;c_y&space;\\&space;0&space;&&space;0&space;&&space;1&space;\end{array}\right]&space;$$">  

where:  

- <img src = "https://latex.codecogs.com/gif.latex?%24%24%20f_x%20%24%24"> , <img src = "https://latex.codecogs.com/gif.latex?%24%24%20f_y%20%24%24">  are the focal lengths along the <img src = "https://latex.codecogs.com/gif.latex?$x$"> and <img src = "https://latex.codecogs.com/gif.latex?$y$"> axis respectively (<img src = "https://latex.codecogs.com/gif.latex?$x$"> is width axis on 2D image, <img src = "https://latex.codecogs.com/gif.latex?$y$"> is height axis on 2D image, <img src = "https://latex.codecogs.com/gif.latex?$z$"> is the orthogonal direction)
- <img src = "https://latex.codecogs.com/gif.latex?%24%24%20c_x%20%24%24"> , <img src = "https://latex.codecogs.com/gif.latex?%24%24%20c_y%20%24%24">  are the pixel coordinates of the principal point on the image plane (ie point aligned with the optical axis of the camera (in our case it is the middle of the image))  

<img src = "https://latex.codecogs.com/gif.latex?%24%24%20f_x%20%24%24"> and <img src = https://latex.codecogs.com/gif.latex?%24%24%20f_y%20%24%24$> can be computed using the horizontal and vertical field of view:  

<img src = "https://latex.codecogs.com/gif.latex?%24%24%20f_x%20%3D%20%5Cdfrac%7Bc_x%7D%7B%5Ctan%20%28FOV_x%20/2%29%7D%24%24"> 
<img src = "https://latex.codecogs.com/gif.latex?%24%24%20f_y%20%3D%20%5Cdfrac%7Bc_y%7D%7B%5Ctan%20%28FOV_y%20/2%29%7D%24%24">

Given a 2D point <img src = " https://latex.codecogs.com/gif.latex?$P$"> with 2D pixel coordinates in the image <img src = " https://latex.codecogs.com/gif.latex?$(u,v)$"> and depth in the camera coordinates <img src = " https://latex.codecogs.com/gif.latex?$z$"> (it can be real depth in meters, and if it is we will recover the metric informaiton, otherwise, the re will be a scale factor), the 3D location of P in the camera coordinates is : 
<img src = "https://latex.codecogs.com/gif.latex?$$&space;\left[\begin{array}&space;{r}x\\&space;y&space;\\&space;z&space;\end{array}\right]&space;=&space;K_c^{-1}&space;\left[\begin{array}&space;{r}uz\\&space;vz&space;\\&space;z&space;\end{array}\right]&space;$$"> 

the multiplication by z on pixel coordinates can be understood with homogeneous coordinates (explanation link : https://www.tomdalling.com/blog/modern-opengl/explaining-homogenous-coordinates-and-projective-geometry/)


If the pitch <img src = " https://latex.codecogs.com/gif.latex?$\epsilon$">of the camera is <img src = " https://latex.codecogs.com/gif.latex?$\neq$"> 0, we can then compute the coordinates in the "real world" coordinate system (not rotated). 
<img src ="https://latex.codecogs.com/gif.latex?$$\left[\begin{array}&space;{r}x'\\&space;y'&space;\\&space;z'&space;\end{array}\right]&space;=&space;\left[\begin{array}&space;{rrr}1&space;&&space;0&space;&&space;0&space;\\&space;0&space;&&space;cos(-\epsilon)&space;&&space;-sin(-\epsilon)&space;\\&space;0&space;&&space;sin(-\epsilon)&space;&&space;cos(-\epsilon)&space;\end{array}\right]&space;\left[\begin{array}&space;{r}x\\&space;y&space;\\&space;z&space;\end{array}\right]&space;$$"> 

For more details on the pinhole camera model, you can refer to this lecture : http://vision.stanford.edu/teaching/cs131_fall1516/lectures/lecture8_camera_models_cs131.pdf
