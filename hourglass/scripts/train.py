import os
from comet_ml import Experiment
import argparse
import torch
import torch.nn as nn
from utils import (
    parsed_args,
    get_config
)
from viz import plot_height_map, plot_from_batch
from trainer import Height_trainer
import data 
from torch.utils.data import DataLoader, Dataset

opts = parsed_args()
config = get_config(opts.config)

root = config['root']
img_train = config['img_train']
img_test = config['img_test']
height_train = config['height_train']
height_test = config['height_test']

comet_key = config['comet_key']
previous_experiment = config['previous_experiment']
project_name = config['project_name']

batch_size = config["batch_size"]
max_iter = config["max_iter"]

display_size = config['display_size']
"""
if config['comet_exp'] :
    if config['previous_experiment'] == "":
        comet_exp = Experiment(comet_key, project_name = project_name)
    else: 
        comet_exp = ExistingExperiment(api_key= comet_key, previous_experiment=config['previous_experiment'])

    comet_exp.log_asset(file_data=opts.config, file_name="config.yaml")
    comet_exp.log_parameter("git_hash", opts.git_hash)
else:
    comet_exp = None
"""    
if config['comet_exp'] :
    comet_exp = True
else:
    comet_exp = None
dataset_train = data.HeightDataset(root, img_train, height_train, transform_params=config['transforms_params'])
dataloader_train = DataLoader(dataset_train, batch_size=batch_size,shuffle=True, num_workers=0)

dataset_test= data.HeightDataset(root, img_test, height_test, transform_params=config['transforms_params'])
dataloader_test = DataLoader(dataset_test, batch_size=batch_size,shuffle=False, num_workers=0)

trainer = Height_trainer(config)
trainer.cuda()

iterations = 0

test_display_images = torch.stack(
    [dataloader_test.dataset[i*100]['image'] for i in range(display_size)]
).cuda()
train_display_images = torch.stack(
    [dataloader_train.dataset[i*100]['image'] for i in range(display_size)]
).cuda()

test_display_mask = torch.stack(
    [dataloader_test.dataset[i*100]['mask'] for i in range(display_size)]
).cuda()
train_display_mask = torch.stack(
    [dataloader_train.dataset[i*100]['mask'] for i in range(display_size)]
).cuda()

epoch = 0
while True:
    print("Epoch %d" % epoch)
    for index, elem in enumerate(dataloader_train):
        inputs, gt = elem['image'].cuda().type(torch.cuda.FloatTensor), elem['mask'].cuda().type(torch.cuda.FloatTensor)
        
        mask = trainer.get_mask_from_val(inputs, 5).type(torch.cuda.FloatTensor)
        
        trainer.model_update(inputs, config, gt, mask, comet_exp, mask_sky=None)
                
        if (iterations + 1) % config["image_save_iter"] == 0:
            with torch.no_grad():
                test_image_outputs = trainer.sample(
                    test_display_images
                )
                train_image_outputs = trainer.sample(
                    train_display_images
                )
                
                
                test_figs = plot_from_batch(test_display_images, test_display_mask,  test_image_outputs)
                train_figs = plot_from_batch(train_display_images, train_display_mask,  train_image_outputs)
                if config['comet_exp']: 
                    trainer.exp.log_image(test_figs, name = "%d_test_heights.jpg" % (iterations))
                    trainer.exp.log_image(train_figs, name = "%d_train_heights.jpg" % (iterations))

        iterations += 1
        if iterations >= max_iter:
            sys.exit("Finish training")
    epoch += 1
