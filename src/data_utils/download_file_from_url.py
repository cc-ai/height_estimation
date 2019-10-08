#!/usr/bin/env python
# coding: utf-8

# In[2]:


from urllib.request import urlretrieve
import os
from zipfile import ZipFile

def download(url, file):
    if not os.path.isfile(file):
        print("Download file... " + file + " ...")
        urlretrieve(url,file)
        print("File downloaded")
        
if __name__ == '__main__':
    download('http://www.cs.cornell.edu/projects/megadepth/dataset/models/best_generalization_net_G.pth',"../MegaDepth/checkpoints/test_local/best_generalization_net_G.pth")
    print("MegaDepth files downloaded")

