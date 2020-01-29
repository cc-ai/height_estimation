import os
import random
import glob
from pathlib import Path

#Create train test txt files 

#path to original images folder
folder_rgb = ""
#path to height maps folder 
folder_height = ""

#where to save the txt files 
train_img_txt = '../data/train_img_sim.txt'
test_img_txt = '../data/test_img_sim.txt'
train_height_txt = '../data/train_height_sim.txt'
test_height_txt = '../data/test_height_sim.txt'

#pattern for file 
img_files = [f for f in os.listdir(folder_rgb) if f.endswith('.png')]
heights = [f for f in os.listdir(folder_height) if f.endswith('.npy')]

img_files.sort()
heights.sort()

data = list(zip(img_files, heights))

random.shuffle(data)

imgs, heights_ = zip(*data)

thresh = 0.75
cut = int(thresh * len(imgs))

with open(train_img_txt, 'w') as file_train:
    for elem in imgs[:cut]:
        file_train.write('%s\n' % elem)

with open(test_img_txt, 'w') as file_test:
    for elem in imgs[cut:]:
        file_test.write('%s\n' % elem)

with open(train_height_txt, 'w') as train_height:
    for elem in heights_[:cut]:
        train_height.write('%s\n' % elem)

with open(test_height_txt, 'w') as test_height:
    for elem in heights_[cut:]:
        test_height.write('%s\n' % elem)