#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import sys
from torch.autograd import Variable
import numpy as np
from MegaDepth.options.train_options import TrainOptions # set CUDA_VISIBLE_DEVICES before import torch
from MegaDepth.data.data_loader import CreateDataLoader
from MegaDepth.models.models import create_model
from skimage import io
from skimage.transform import resize
from PIL import Image
import os


def test_simple(model, img_path, size, save_path):
    print(size)
    input_width, input_height = size
    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()

    img = np.float32(io.imread(img_path))/255.0
    img = resize(img, (input_height, input_width), order = 1)
    input_img =  torch.from_numpy( np.transpose(img, (2,0,1)) ).contiguous().float()
    input_img = input_img.unsqueeze(0)

    input_images = Variable(input_img.cuda() )
    pred_log_depth = model.netG.forward(input_images) 
    pred_log_depth = torch.squeeze(pred_log_depth)
    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)
    save_filename = os.path.splitext(os.path.basename(img_path))[0] + '_depth'
    io.imsave(os.path.join(save_path, save_filename + '.jpg'), pred_inv_depth)
    np.save(os.path.join(save_path, save_filename + '.npy'), pred_inv_depth)
    print("saved to files "+ os.path.join(save_path, save_filename))
    Image.open(os.path.join(save_path, save_filename)).show()
    return(os.path.join(save_path, save_filename))
  #  sys.exit()

def get_depthmap_img(img_path, save_path, size, checkpoints_dir):
    opt = TrainOptions()
    #print(opt)
    opt.gpu_ids = '0,1'
    opt.isTrain = True
    opt.checkpoints_dir = checkpoints_dir
    opt.name = 'test_local/'
    model = create_model(opt)
    path = test_simple(model, img_path, size,save_path)
    
    print("We are done")
    return(path)

 def default_transforms():
    flip = torchvision.transforms.functional.hflip
    color = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.4, saturation=0.2, hue=0)

    transforms_ = [flip, torchvision.transforms.Compose([flip, color]), color, color]
    return(transforms_)


def test_nimage(model, img_path, size, save_path, transforms_):
    nimage = len(transforms_ + 1)
    print(size)
    input_width, input_height = size
    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()
    img_arrays = np.array([resize(np.float32(im)/255.0, (input_height, input_width), order = 1) for im in images])

    input_imgs =  torch.from_numpy( np.transpose(img_arrays, (0,3,1,2)) ).contiguous().float()


    input_images = Variable(input_imgs.cuda() )
    with torch.no_grad():
        pred_log_depth = model.netG.forward(input_images) 
    pred_log_depth = torch.squeeze(pred_log_depth)
    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

    pred_inv_depth[1:3] = np.flip(pred_inv_depth[1:3], 2)

    for i, im in enumerate(pred_inv_depth):
        pred_inv_depth[i] -= np.min(im)
        pred_inv_depth[i] /= np.max(pred_inv_depth[i])

    for i in range(nimages):
        save_filename = os.path.splitext(os.path.basename(img_file))[0] + '_depth' + str(i) + '.jpg'
        io.imsave(os.path.join(save_path, save_filename), pred_inv_depth[i])
        print("Image saved to "+ os.path.join(save_path, save_filename))
        Image.open(os.path.join(save_path, save_filename)).show()

    mean_pred = np.mean(pred_inv_depth, axis = 0)

    save_filename = os.path.splitext(os.path.basename(img_file))[0] + 'meanpred_depth'  + '.jpg'
    io.imsave(os.path.join(save_path, save_filename), mean_pred)
    print("Image saved to "+ os.path.join(save_path, save_filename))
    Image.open(os.path.join(save_path, save_filename)).show()
    
    
    
def test_nimage_npy(model, img_path, size, save_path, transforms_, indices_flip):
   """
   transforms_: list of tranformations to apply. The model will run on the original image + copies of the original transformed by transforms_. In total the batch will be of size transforms_+1
   indices_flip: indices of the transforms that have flip (vertical)(if transforms[0] has flip, the index indicated will be 1)
   """
    nimage = len(transforms_ + 1)
    print(size)
    input_width, input_height = size
    total_loss =0 
    toal_count = 0
    print("============================= TEST ============================")
    model.switch_to_eval()
    img_arrays = np.array([resize(np.float32(im)/255.0, (input_height, input_width), order = 1) for im in images])

    input_imgs =  torch.from_numpy( np.transpose(img_arrays, (0,3,1,2)) ).contiguous().float()


    input_images = Variable(input_imgs.cuda() )
    with torch.no_grad():
        pred_log_depth = model.netG.forward(input_images) 
    pred_log_depth = torch.squeeze(pred_log_depth)
    pred_depth = torch.exp(pred_log_depth)

    # visualize prediction using inverse depth, so that we don't need sky segmentation (if you want to use RGB map for visualization, \
    # you have to run semantic segmentation to mask the sky first since the depth of sky is random from CNN)
    pred_inv_depth = 1/pred_depth
    pred_inv_depth = pred_inv_depth.data.cpu().numpy()
    # you might also use percentile for better visualization
    pred_inv_depth = pred_inv_depth/np.amax(pred_inv_depth)

    pred_inv_depth[indices_flip] = np.flip(pred_inv_depth[1:3], 2)

    for i, im in enumerate(pred_inv_depth):
        pred_inv_depth[i] -= np.min(im)
        pred_inv_depth[i] /= np.max(pred_inv_depth[i])

    mean_pred = np.mean(pred_inv_depth, axis = 0)

    save_filename = os.path.splitext(os.path.basename(img_file))[0] + 'meanpred_depth'
    io.imsave(os.path.join(save_path, save_filename + '.jpg'), mean_pred)
    np.save(os.path.join(save_path, save_filename + '.npy'), mean_pred )
    print("Image saved to "+ os.path.join(save_path, save_filename))
   
if __name__ == "__main__":
    img_path = '../DataSV/milacropresize.jpg'
    save_path = '.'
    size = [512,512]
    opt = TrainOptions().parse()
    model = create_model(opt)
    test_simple(model, img_path, size,save_path)
    print("We are done")


