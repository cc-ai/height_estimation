#!/usr/bin/env python
# coding: utf-8

# In[15]:


import google_streetview.api as gsv_api
import google_streetview.helpers as gsv_helpers
import numpy as np 
import os

def param_block(address, key):
    """
    make a param query in the GSV API format
    
    """
    param = {
        "size": "512x512",
        "pitch": "0",
        "roll": "0",
        "radius": "1000",
        "key": str(key),
        "source": "outdoor",
        "fov": "120",
        'location':str(address[0]) + ',' + str(address[1]),
    }
    return param

def get_image_from_gps_adresses(params):
    """
    Get a result object containing the link to an street view image
    in the vicinity of the specified lat and long coordinates
    
    use as results.download_links(".") 
    change "." to another path not to download in cwd
    ex: 
    res = get_image_from_gps("45.5307147,-73.6157818")
    res.download_links("~/Downloads") 
    
    """
    api_list = gsv_helpers.api_list(params)
    results = gsv_api.results(api_list)
    return (results)

def get_images(params, folder_path):
    """
    save the images
    Inputs: 
        params : list of 
        folder_path: path of the folder where to save the images, must end with '/'
    """
    paths = []
    for index, elem in enumerate(params) :     
        results = get_image_from_gps_adresses(elem)
        metadata_file = 'metadata%d.json' % index
        results.download_links(folder_path)
        os.rename(folder_path + 'gsv_0.jpg',folder_path + 'gsv_'+'{:06d}'.format(index)+'.jpg')
        paths.append(folder_path + 'gsv_'+'{:06d}'.format(index)+'.jpg')
    print("All images fetched")
    return (paths)
    

