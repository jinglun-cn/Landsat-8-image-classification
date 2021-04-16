# -*- coding: utf-8 -*-

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
from rasterio.warp import calculate_default_transform, reproject, Resampling
from get_wrs import ConvertToWRS

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


def cal_path_row(site):
    '''Calculate Path/Row for each site'''
    conv = ConvertToWRS(shapefile='./WRS2_descending/WRS2_descending.shp')
    path_row = conv.get_wrs(site['Latitude'], site['Longitude'])[0]  #conv.get_wrs(lat, lon)  
    site['path'] = path_row['path']
    site['row'] = path_row['row']
    return site


# Form a bulk of L8 products to download

def form_bulk(bulk_list, sites, s3_scenes):  

    # Iterate over sites to select a bulk of scenes
    for index, site in sites.iterrows():

        # check if the site is covered in previous scenes
        covered = False
        for scene in bulk_list:
            if (scene.path == site.path and scene.row == site.row):              
                sites.loc[index, 'productId'] = scene.productId            
                covered = True  
                
        if not covered:

            # Filter the Landsat S3 table for images matching path/row, cloudcover and processing state.
            scenes = s3_scenes[(s3_scenes.path == site.path) & (s3_scenes.row == site.row) & 
                              (s3_scenes.cloudCover <= 5) & 
                              (~s3_scenes.productId.str.contains('_T2')) &
                              (~s3_scenes.productId.str.contains('_RT')) &
                              (s3_scenes.acquisitionDate.str.contains('2016-'))]
            # print(' Found {} images\n'.format(len(scenes)))

            # If any scene exists, select the one that have the minimum cloudCover.
            if len(scenes):
                scene = scenes.sort_values('cloudCover').iloc[0]        
                sites.loc[index, 'productId'] = scene.productId 

                # Add the selected scene to the bulk download list.
                bulk_list.append(scene)      
            else:
                print('cannot find a scene for the site ID={}'.format(site.ID))  
    return bulk_list              

def download_and_stack_product(row_bulk_frame):
    '''   Download and stack Landsat8 bands   '''
    LANDSAT_PATH = './Landsat8data/'


    entity_dir = os.path.join(LANDSAT_PATH, row_bulk_frame.productId)
    # if this dir exists, assume data are downloaded
    if not os.path.exists(entity_dir):
        # Print some the product ID
        print('\n', 'Downloading L8 data:', row_bulk_frame.productId)
        # print(' Checking content: ', '\n')

        # Request the html text of the download_url from the amazon server. 
        # download_url example: https://landsat-pds.s3.amazonaws.com/c1/L8/139/045/LC08_L1TP_139045_20170304_20170316_01_T1/index.html
        response = requests.get(row_bulk_frame.download_url)

        # If the response status code is fine (200)
        if response.status_code == 200:

            # Import the html to beautiful soup
            html = BeautifulSoup(response.content, 'html.parser')

            # Create the dir where we will put this image files.
            entity_dir = os.path.join(LANDSAT_PATH, row_bulk_frame.productId)
            os.makedirs(entity_dir, exist_ok=True)

            # Second loop: for each band of this image that we find using the html <li> tag
            for li in html.find_all('li'):

                # Get the href tag
                file = li.find_next('a').get('href')

                # print('  Downloading: {}'.format(file))

                response = requests.get(row_bulk_frame.download_url.replace('index.html', file), stream=True)

                with open(os.path.join(entity_dir, file), 'wb') as output:
                    shutil.copyfileobj(response.raw, output)
                del response

    # Stack bands 1-7,9

    # Obtain the list of bands 1-7,9
    landsat_bands = glob(os.path.join(entity_dir, '*B[0-7,9].TIF'))       
    landsat_bands.sort()

    # Read metadata of first file
    with rio.open(landsat_bands[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    meta.update(count = len(landsat_bands))

    # Read each layer and write it to stack
    stackfile = os.path.join(entity_dir, row_bulk_frame.productId+'_Stack.TIF')
    # print('Stacking L8 bands: {}'.format(row_bulk_frame.productId))
    with rio.open(stackfile, 'w', **meta) as dst:
        for id, layer in enumerate(landsat_bands, start=1):
            with rio.open(layer) as src1:
                dst.write_band(id, src1.read(1))

    # Reprojecting
    # print('Reprojecting...')
    reproject_ras(inpath = stackfile, 
                  outpath = stackfile, 
                  new_crs = 'EPSG:4326')    
          


def crop_rectangle(lat, lon, image_width, image_height, res, in_file, out_file = './out.TIF'):
    '''crop a rectangle around a point in Lat/Lon CRS'''

    with rio.open(in_file) as src:

        # CRS transformation
        src_crs = rio.crs.CRS.from_epsg(4326) # latlon crs
        dst_crs = src.crs # current crs
        xs = [lon] 
        ys = [lat] 
        coor_transformed = warp.transform(src_crs, dst_crs, xs, ys, zs=None)
        coor = [coor_transformed[0][0], coor_transformed[1][0]]
        # print('coor: ', coor )

        # Returns the (row, col) index of the pixel containing (x, y) given a coordinate reference system
        py, px = src.index(coor[0], coor[1])

        # Build window with right size
        p_width = image_width//res
        p_height = image_height//res        
        window = rio.windows.Window(px - p_width//2, py - p_height//2, p_width, p_height)
        # print('window: ', window)

        # Read the data in the window
        clip = src.read(window=window)
        # print('clip: ', clip)

        # write a new file
        meta = src.meta
        meta['width'], meta['height'] = p_width, p_height
        meta['transform'] = rio.windows.transform(window, src.transform)

        with rio.open(out_file, 'w', **meta) as dst:
            dst.write(clip)                

def stack_rasters(raster_list, outfile):    

    # Read metadata of a certain file
    with rio.open(raster_list[0]) as src0:
        meta = src0.meta

    # Update meta to reflect the number of layers
    num_layers = 0
    for f in raster_list:
        with rio.open(f) as src:
            num_layers += src.count
    meta.update(count = num_layers)

    # Read each layer and write it to stack
    # print('  Stacking site_ID: {}'.format(site.ID))
    with rio.open(outfile, 'w', **meta) as dst:
        stack_band_offset = 0
        for id, f in enumerate(raster_list, start=1):
            with rio.open(f) as src:
                for band_f in range(1, src.count+1):
                    # Cast dtype, matching L8
                    band_to_write = src.read(band_f).astype(np.uint16)
                    dst.write_band(stack_band_offset+band_f, band_to_write)    
                stack_band_offset += src.count

def tif_to_np(tif_file):
    with rio.open(tif_file) as src:
        return src.read() 


def classify(arr_l8, model, patch_size):
    ''''arr_l8 is a 4D array of size (N, X, Y, B) '''
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
    
    return arr_cls            
