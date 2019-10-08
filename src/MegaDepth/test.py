#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import sys
from torch.autograd import Variable
import numpy as np
from src.MegaDepth.options.train_options import TrainOptions # set CUDA_VISIBLE_DEVICES before import torch
from src.MegaDepth.data.data_loader import CreateDataLoader
from src.MegaDepth.models.models import create_model
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
    save_filename = os.path.splitext(os.path.basename(img_path))[0] + '_depth.jpg'
    io.imsave(os.path.join(save_path, save_filename), pred_inv_depth)
    print("Image saved to "+ os.path.join(save_path, save_filename))
    Image.open(os.path.join(save_path, save_filename)).show()
  #  sys.exit()

def get_depthmap_img(img_path, save_path, size):
    opt = TrainOptions()
    #print(opt)
    opt.gpu_ids = '0,1'
    opt.isTrain = True
    opt.checkpoints_dir = '/network/home/tengmeli/src/MegaDepth/checkpoints/'
    opt.name = 'test_local/'
    model = create_model(opt)
    test_simple(model, img_path, size,save_path)
    print("We are done")
    

                     
if __name__ == "__main__":
    img_path = '../DataSV/milacropresize.jpg'
    save_path = '.'
    size = [512,512]
    opt = TrainOptions().parse()
    model = create_model(opt)
    test_simple(model, img_path, size,save_path)
    print("We are done")


