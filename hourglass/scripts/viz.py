import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_height_map(inputs, labels, outputs):
    """
    create an image of original image (inputs) + ground truth height map + predicted height map
    dimensions Channels * H * W
    """
    label_img = labels[0,0,:,:].cpu().type(torch.FloatTensor)
    output_img = outputs[0,0,:,:].detach().cpu().type(torch.FloatTensor)
    heights = torch.cat([label_img, output_img], dim = 1)
    fig = plt.figure(figsize = (9,3))
    gs = gridspec.GridSpec(1,2, width_ratios=[1, 2.13])
    gs.update(wspace=0., hspace=0.)
    ax = plt.subplot(gs[0,1])
    plt.axis('off')
    im = ax.imshow(heights, cmap=plt.cm.RdBu, interpolation='bilinear')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    ax2 = plt.subplot(gs[0,0])#(121)
    plt.axis('off')
    ax2.imshow(inputs[0,:,:,:].permute(1,2,0).cpu())
    plt.tight_layout()
    return(fig)

def fig2data(fig):
    """
    Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    Input: fig a matplotlib figure
    return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = (h, w,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def combine_images(figs):
    """
    Input: list of np.arrays of images to combine
    """
    stacked_images = np.vstack(figs)
    return(stacked_images)

def plot_from_batch(inputs, labels, outputs, resize_factor = 1):
    """
    Dimension of entry tensors : Batch_size * Channels * H * W
    """
    figures = []
    for index, elem in enumerate(inputs):
        figures.append(plot_height_map(inputs, labels, outputs))
    figsarray = [fig2data(elem) for elem in figures]
    figs = combine_images(figsarray)                   
    return(figs)