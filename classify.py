'''classify locations using pretrained model'''

import os, shutil, sys, time
import numpy as np
import pandas as pd
from glob import glob
import rasterio as rio
from rasterio import warp
import zipfile
import requests
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import tensorflow as tf


model_path = './pretrained.hdf5'
sites_path = './sites_classify.csv'
LANDSAT_PATH = './Landsat8data/'
for PATH in [LANDSAT_PATH]:
    os.makedirs(PATH, exist_ok=True)

# unit: m
image_width = 3840
image_height = 3840
res = 30    

# get input sites, coordinates in LatLong format 
sites = pd.read_csv(sites_path, header=0)

# Form downloading bulk frame
t_start = time.time()

# Read scenes frame from Amazon s3
s3_scenes = pd.read_csv('http://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz', compression='gzip')

# conversion function: LatLon to PathRow 
# from get_wrs import ConvertToWRS
# conv = ConvertToWRS(shapefile='./WRS2_descending/WRS2_descending.shp')
# usage
# conv.get_wrs(25.411914, -80.496381)  # conv.get_wrs(lat, lon)  

from util import cal_path_row, form_bulk, download_and_stack_product, crop_rectangle, tif_to_np

sites = sites.apply(lambda r : cal_path_row(r), axis=1)

bulk_list = []
bulklist = form_bulk(bulk_list, sites, s3_scenes)  

# Check selected images
bulk_frame = pd.concat(bulk_list, 1).T
bulk_frame.head()

t_end = time.time()
print ("Time elapsed: {} s".format(t_end - t_start))

t_start = time.time()
# For each productID
for i, row_bulk_frame in bulk_frame.iterrows(): 
    download_and_stack_product(row_bulk_frame)
    # For each site with specified productID

    # Crop each site
    # print('Cropping sites in {}...'.format(row_bulk_frame.productId))
    for index, site in sites.iterrows():
        if site.productId == row_bulk_frame.productId:          
              L8_file = os.path.join(LANDSAT_PATH, 'L8_Site_ID_{}_LARGE.TIF'.format(site.ID))

              # Crop L8
              in_dir = os.path.join(LANDSAT_PATH, site.productId)
              in_file = os.path.join(in_dir, site.productId+'_Stack.TIF') 
              crop_rectangle(site.Latitude, site.Longitude, image_width, 
                             image_height, res, in_file, L8_file)
              
t_end = time.time()
print ("Time elapsed: {} s".format(t_end - t_start))

# dataset is a 4d np-array: (sample_index, band, x-coor, y-coor)
arr_l8 = np.array([tif_to_np(L8_file.format(site.ID)) for i, site in sites.iterrows()])
arr_l8 = arr_l8.astype('float32')

# make channel_last
arr_l8 = np.swapaxes(arr_l8, 1, 3)


# patch_size must be odd
patch_size = 15
model = tf.keras.models.load_model(model_path)

s = patch_size
N, X, Y, B = arr_l8.shape[0], arr_l8.shape[1], arr_l8.shape[2], arr_l8.shape[3]
# Pixel (x, y) in sample n is the center of patches[m]
# m= n*(X-s+1)*(Y-s+1) + (y-2)*(X-s+1) + (x-2), x,y,n starts from 0

# extract patches
patches = []
for n in range(N):
    for y in range(Y-s+1):         
        for x in range(X-s+1):
                # patch = arr_l8[n, x:x+s, y:y+s, :].copy()
                # cls = np.argmax(model.predict(patch[np.newaxis, ...]), axis=-1)[0]
                # arr_cls[n, x+s//2, y+s//2] = cls
                patches.append(arr_l8[n, x:x+s, y:y+s, :])

patches = np.array(patches)
labels = np.argmax(model.predict(patches), axis=-1)

# classification array
arr_cls = np.zeros(arr_l8.shape[:-1])

i = 0
for n in range(N):
    for y in range(Y-s+1):         
        for x in range(X-s+1):
            arr_cls[n, x+s//2, y+s//2] = labels[i]
            i += 1

# output results to a 4D numpy array
np.save('./arr_cls.npy', arr_cls)



