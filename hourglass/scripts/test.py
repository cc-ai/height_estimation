from __future__ import print_function
from utils import get_config
import argparse
from trainer import Height_trainer
from torch.autograd import Variable
import torchvision.utils as vutils
import os
import torch
from viz import plot_height_map, plot_from_batch_v2
import data 
from torch.utils.data import DataLoader, Dataset
import sys
from torchvision import transforms
from PIL import Image
import glob
import tqdm as tq

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help="network configuration file")
parser.add_argument("--input", type=str, help="directory of input images")
parser.add_argument("--output_folder", type=str, help="output image directory")
parser.add_argument("--checkpoint", type=str, help="checkpoint of generator")

opts = parser.parse_args()

# Create output folder if it does not exist
if not os.path.exists(opts.output_folder):
    os.makedirs(opts.output_folder)

# Load experiment setting
config = get_config(opts.config)
trainer = Height_trainer(config)

try:
    ckpt = torch.load(opts.checkpoint)
    trainer.model.load_state_dict(ckpt)#chkpt['model_state_dict'])
except:
    sys.exit("Cannot load the checkpoints")

# Send the trainer to cuda
trainer.cuda()
trainer.eval()


# Define the list of non-flooded images
list_ims = sorted(glob.glob(opts.input+'*'))

# Assert there are some elements inside
if len(list_ims) ==0:
    sys.exit('Image list is empty. Please ensure opts.input ends with a /')

# Define the transform to infer on the images (the image has to fit in the hourglass model)    
transform = transforms.Compose([
    transforms.Resize(config['transforms']['resize_size']),
    transforms.CenterCrop(config['transforms']['crop_size']),
    transforms.ToTensor(),        
                ])            
# Inference
with torch.no_grad():
    for j in tq.tqdm(range(len(list_ims))):     
        # Define image path
        path_im = list_ims[j]
        
        # Load and compute the height map of the image
        im  = Variable(
            transform(Image.open(path_im).convert("RGB"))
        )        
        
        if config['save_input']:
            im.permute(1,2,0).numpy().astype(np.uint8).save(os.path.join(output_folder, "input{:03d}.jpg".format(j)))
        im = im.unsqueeze(0)
        outputs = trainer.model(im.cuda())
        
        outputs.data.cpu().numpy().save(os.path.join(output_folder, "output{:03d}.npy".format(j)))
        # Define output path
        path = os.path.join(output_folder, "output{:03d}.jpg".format(j))
        
        if config['gt_root'] is not None:
            gt = os.path.join(config['gt_root'], os.path.basename(path_im)[:-3] + 'npy')
            fig = plot_from_batch_v2(im, torch.tensor(np.load(gt)).unsqueeze(0).unsqueeze(0), outputs)
        
        # Save image 
            fig.save(path)
