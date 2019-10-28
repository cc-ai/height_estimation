import numpy as np
import matplotlib.pyplot as plt
from math import tan, sin, cos, pi, atan, radians, degrees
import matplotlib.pyplot as plt

import skimage
from PIL import Image
from open3d import *

from pathlib import Path

# load image and depth map
def get_ratio(flood, img):
    return len(flood[0]) / (img.shape[0] * img.shape[1])


def color(img, flood, paint=[114, 139, 161]):
    """paint an image from the indices in flood

    Args:
        img (np.array): image to paint
        flood (np.where): where to flood
        paint (list, optional): Color to paint with.

    Returns:
        PIL.Image: painted image
    """
    colors = img.copy().reshape(512 * 512, 3)
    colors[flood, :] = paint
    flooded_im = colors.reshape(512, 512, 3)
    return Image.fromarray(flooded_im)


def get_cood_threshold(threshold, coords_proj_):
    """Scale a [0; 1] threshold to the scale of coords_proj_

    Args:
        threshold (float): [0;1] float ratio
        coords_proj_ (np.array): non-metric projected coordinates

    Returns:
        float: scaled threshold
    """
    return coords_proj_.min() + threshold * (coords_proj_.max() - coords_proj_.min())


def get_flood_idx(coords_proj_, coor_threshold):
    """get the flattened list of indices where the images should be painted in blue

    Args:
        coords_proj_ (np.array): projected coordinats
        coor_threshold (float): water-level threshold (non-metric)

    Returns:
        tuple: np.where output
    """
    return np.where(coords_proj_[:, 1] < coor_threshold)


def makegif(
    coords_proj_, img, img_path, max_ratio=0.5, step=1e-3, duration=3, zero_ratios_len=5
):
    img_path = Path(img_path)
    images = []
    max_ratio = 0.5
    threshold = 0

    while threshold < 1:
        # flood everything under threshold
        coor_threshold = get_cood_threshold(threshold, coords_proj_)
        flood = get_flood_idx(coords_proj_, coor_threshold)
        ratio = get_ratio(flood, img)
        if ratio == 0:
            threshold += step
            continue
        if ratio > max_ratio:
            break
        images.append(color(img, flood))
        threshold += step

    images = [Image.fromarray(img)] * zero_ratios_len + images

    gif_name = "{}_{}_{}.gif".format(str(img_path.parent / img_path.stem), step, max_ratio)
    print(gif_name)
    images[0].save(
        gif_name,
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=1000 * duration / len(images),
        loop=1,
    )


def make_gifs(img_path, depth_path, max_ratio, step, duration, zero_ratios_len):
    """Store a gif of an image, with zero_ratios_len images without flood before flooding the image
    one step at a time, until max_ratio

    Args:
        img_path (Path): where to find the image
        depth_path (Path): where to find its
        max_ratio (float): flood until max_ratio * 100 % of the image is flooded
        step (float): how fine the grid search is
        duration (int): final gif length in seconds
        zero_ratios_len (int): how many frames to put without floods, at the begining of the gif

    Returns:
        None: saves gif to disk
    """

    img = np.array(Image.open(img_path))
    depth = np.array(Image.open(depth_path))

    # display images side by side
    Image.fromarray(
        np.hstack((img, np.array(Image.open(depth_path).convert("RGB"))))
    ).resize((600, 300))

    # inverse depth map
    depth_ = 1 / (depth / 255)
    depth_ /= np.max(depth_)
    depth_ *= 255

    np.min(depth_), np.max(depth_)

    # The initial FOV in the API was 120°. But we center-cropped the image, taking out margins of 20 pixels on all sides (to get rid of the watermarks)
    # We then resized the image to 512*512 so that we could apply MegaDepth on it ( some pixels are dilated)

    H, W = 512, 512
    init_FOVx = pi / 3  # half FOV
    init_FOVy = pi / 3  # half FOV
    FOVx = 2 * atan(
        tan(init_FOVx) * ((W / 2) - 20) / (W / 2)
    )  # initial FOV - angle corresponding to the crop
    FOVy = 2 * atan(((H / 2) - 20) / (H / 2) * tan(init_FOVy))

    cx = W / 2
    cy = H / 2

    fx = cx / tan(FOVx / 2)
    fy = cy / tan(FOVy / 2)

    Kc = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    inv_Kc = np.linalg.inv(Kc)

    def my_func(a):
        return inv_Kc.dot(a)

    coord1, coord2 = np.meshgrid(range(H), range(W))
    coords_plane = np.stack((coord1.flatten(), coord2.flatten()), axis=-1)
    coords3 = np.append(coords_plane, -np.expand_dims(depth_.flatten(), axis=1), axis=1)
    # minus sign on the third direction because the pinhole model inverses
    coords3[:, 0] = coords3[:, 0] * coords3[:, 2]
    coords3[:, 1] = coords3[:, 1] * coords3[:, 2]

    # get the coordinates in the camera coordinate system
    coords_proj = np.apply_along_axis(my_func, 1, coords3)

    # apply rotation matrix
    epsilon = radians(10)
    # this is the rotation matrix from the non rotated to rotated (camera) coordinate system
    rotation = np.array(
        [[1, 0, 0], [0, cos(epsilon), -sin(epsilon)], [0, sin(epsilon), cos(epsilon)]]
    )
    # we actually right apply the transpose of the rotation matrix  which is the same as left applying the matrix from camera coordinate system to real
    coords_proj_ = np.dot(coords_proj, rotation)

    makegif(coords_proj_, img, img_path, max_ratio, step, duration, zero_ratios_len)


if __name__ == "__main__":

    """For each image in base/, this procedure creates a gif illustrating the image's flooding
    process. /!\ images must be in ".jpg" extension, and paired (base/img.jpg, base/img_depth.jpg)

    Each image has a different range of values due to the noise in the scene reconstruction.
    So we normalise the threshold search to [0; 1]: each pixels whose height is:

        < coords.min() + threshold * (coords.max() - coords.min())

    Is going to be flooded.
    We search for thresholds as follows:
        * compute un-normalized threshold
        * compute ratio of flooded pixels wrt number of pixels
        * if ratio == 0: continue
            * and store threshold for later use in non-flooded initial frames
        * if ratio > max_ratio (typically max_ratio = 0.5 => 50% of pixels flooded)
            * break
            * prepend latest 5 thresholds with ratio == 0
            * >> save gif as "{imgdir}/{img_name}.gif"
        * else: (interesting case)
            * "paint" (overwrite) pixels < threshold in blue
            * store resulting image
    """

    # Where to find images in formats img.jpg, img_depth.jpg
    base = "/Users/victor/Downloads/gsv_000000"
    # max image ratio to flood (num flooded pixels / num non_flooded)
    max_ratio = 0.5
    # Grid search precision
    step = 1e-3
    # Total resulting gif duration
    duration = 3
    # How many non-flooded frames to prepend to the gif
    zero_ratios_len = 5

    imgs = filter(lambda p: "depth" not in p.name, Path(base).glob("*.jpg"))

    for i in imgs:
        d = i.parent / (i.stem + "_depth.jpg")
        make_gifs(i, d, max_ratio, step, duration, zero_ratios_len)
