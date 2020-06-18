# -*- coding: utf-8 -*-

# !pip install rasterio
# !pip install shapely

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
from util import cal_path_row, form_bulk, download_and_stack_product, crop_rectangle, stack_rasters, tif_to_np



"""Download data"""

NLCD_PATH = './NLCD/'
LANDSAT_PATH = './Landsat8data/'
L8_NLCD_PATH = './L8_NLCD/'
for PATH in [NLCD_PATH, LANDSAT_PATH, L8_NLCD_PATH]:
    os.makedirs(PATH, exist_ok=True)

L8_NLCD_file = os.path.join(L8_NLCD_PATH,'L8_NLCD_Site_ID_{}_LARGE.TIF')

# unit: m
image_width = 3840
image_height = 3840
res = 30    

# get input sites, coordinates in LatLong format 
sites = pd.read_csv('./sites_train.csv', header=0)

# Download and unzip NLCD dataset
# This may take more than 5 minutes
t_start = time.time()

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

print('Downloading NLCD...')
download_url('https://s3-us-west-2.amazonaws.com/mrlc/NLCD_2016_Land_Cover_L48_20190424.zip',
             os.path.join(NLCD_PATH, 'NLCD.zip'))     

print('Unzipping NLCD...')
with zipfile.ZipFile(os.path.join(NLCD_PATH, 'NLCD.zip'), 'r') as zip_ref:
    zip_ref.extractall(NLCD_PATH)       


# Reproject NLCD. Takes ~15min

NLCD_FILE = 'NLCD_2016_Land_Cover_L48_20190424.img'
NLCD_reprojected = 'NLCD_reprojected.tif'

from rasterio.warp import calculate_default_transform, reproject, Resampling

def reproject_ras(inpath, outpath, new_crs):
    dst_crs = new_crs # CRS for web meractor 

    with rio.open(inpath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        with rio.open(outpath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rio.band(src, i),
                    destination=rio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)

inpath = os.path.join(NLCD_PATH, NLCD_FILE)
outpath = os.path.join(NLCD_PATH, NLCD_reprojected)


print('Reprojecting...')
reproject_ras(inpath = inpath, 
             outpath = outpath, 
             new_crs = 'EPSG:4326') # match Lon/Lat CRS

t_end = time.time()

print ("Time elapsed: {} s".format(t_end - t_start))

# Form downloading bulk frame

t_start = time.time()

# Read scenes frame from Amazon s3
s3_scenes = pd.read_csv('http://landsat-pds.s3.amazonaws.com/c1/L8/scene_list.gz', compression='gzip')

# conversion function: LatLon to PathRow 
# from get_wrs import ConvertToWRS
sites = sites.apply(lambda r : cal_path_row(r), axis=1)


# Form a bulk of L8 products to download
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
              NLCD_file = os.path.join(NLCD_PATH, 'NLCD_Site_ID_{}_LARGE.TIF'.format(site.ID))

              # Crop L8
              in_dir = os.path.join(LANDSAT_PATH, site.productId)
              in_file = os.path.join(in_dir, site.productId+'_Stack.TIF') 
              crop_rectangle(site.Latitude, site.Longitude, image_width, 
                             image_height, res, in_file, L8_file)
              # Crop NLCD
              in_file = os.path.join(NLCD_PATH, NLCD_reprojected) 
              crop_rectangle(site.Latitude, site.Longitude, image_width, 
                             image_height, res, in_file, NLCD_file) 
              
              # Stack L8 and NLCD
              raster_list = [L8_file, NLCD_file]
              stack_rasters(raster_list, L8_NLCD_file.format(site.ID))

    # Delete Landsat8 product raw data to save space (optional) 
    # shutil.rmtree(os.path.join(LANDSAT_PATH, row_bulk_frame.productId))     

t_end = time.time()
print ("Time elapsed: {} s".format(t_end - t_start))

L8_NLCD_file = os.path.join(L8_NLCD_PATH,'L8_NLCD_Site_ID_{}_LARGE.TIF')
       
# dataset is a 4d np-array: (sample_index, band, x-coor, y-coor)
dataset = np.array([tif_to_np(L8_NLCD_file.format(site.ID)) for i, site in sites.iterrows()])

# swap axes to ensure channel_last
arr = np.swapaxes(dataset, 1, 3)

# drop NaN
arr_new = []
for i in range(arr.shape[0]):
    if not 0 in arr[i,8,:,:]:
        arr_new.append(arr[i,:,:,:])
arr_new = np.array(arr_new)
arr = arr_new

np.save('./L8_NLCD_extracted_dataset.npy', arr)

# # Visualization

# arr = np.load('./L8_NLCD/L8_NLCD_extracted_dataset_blast.npy')

# # arr[9][4] represents the 10th site, 5th band
# # Visualize
# plt.imshow(arr[9][4])
# plt.show()

# print('Shape of the image: ', arr[9][4].shape)
# print(arr[9][4])

# # Access the pixel (40, 80)

# print('Pixel value: ', arr[9][4][40][80])
