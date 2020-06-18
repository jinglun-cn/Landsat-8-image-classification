# -*- coding: utf-8 -*-
"""
LAS file processing
"""

# !pip install rasterio
# !pip install laspy

import glob, os, zipfile
import numpy as np
import rasterio as rio
from rasterio import warp
import laspy

las_path = '007.zip'
L8_NLCD_path = 'L8_NLCD.zip'

def prep_file(las_path, L8_NLCD_path, siteID=6):
    '''extract data and get las_list and the required tif file for L8_NLCD'''

    # prep las
    las_filename, ext = os.path.splitext(las_path)
    with zipfile.ZipFile(las_path, 'r') as zip_ref:
        zip_ref.extractall('./')
    las_zips = [file for file in glob.glob(os.path.join(las_filename, "*.zip"))]
    for zip_file in las_zips:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(las_filename))
    las_list = [file for file in glob.glob(os.path.join(las_filename, "*.las"))]

    # prep L8_NLCD
    with zipfile.ZipFile(L8_NLCD_path, 'r') as zip_ref:
        zip_ref.extractall('./')
    L8_NLCD_tif = os.path.join('L8_NLCD', 'L8_NLCD_Site_ID_{}_LARGE.TIF'.format(siteID))

    return las_list, L8_NLCD_tif

def get_elevation(lat, lon, las_list, las_crs=2881):
    '''get elevation data around a given coor'''

    src_crs = rio.crs.CRS.from_epsg(4326) # latlon crs
    dst_crs = rio.crs.CRS.from_epsg(las_crs) # las file crs in epsg
    xs = [lon]
    ys = [lat]
    coor_transformed = warp.transform(src_crs, dst_crs, xs, ys, zs=None)
    x, y = coor_transformed[0][0], coor_transformed[1][0]

    # search las files for (x, y)
    z_avg = 0 # output 0 if missing
    square_size = 10 # in NAD83(HARN), ~10 meters/feet?
    for las_file in las_list:
        inFile = laspy.file.File(las_file, mode = "r")
        if inFile.header.min[0] < x and x < inFile.header.max[0] \
            and inFile.header.min[1] < y and y < inFile.header.max[1]:

            coor = np.vstack((inFile.x, inFile.y, inFile.z)).transpose()
            # focus on a small square to calculate z
            X_valid = np.logical_and((x-square_size//2 < coor[:,0]), (x+square_size//2 > coor[:,0]))
            coor = coor[np.where(X_valid)]
            Y_valid = np.logical_and((y-square_size//2 < coor[:,1]), (y+square_size//2 > coor[:,1]))
            coor = coor[np.where(Y_valid)]

            if len(coor[:,2]) == 0: # dealing with missing data
                z_avg = 0
            else:
                z_avg = coor[:,2].mean()
        inFile.close()
    return z_avg

def classify_z(z):
    cls = 0 # low-rise
    if 15 < z and z <= 40:
        cls = 1 # mid-rise
    if z > 40:
        cls = 2 # high-rise
    return cls

def get_site_elevation(las_path, L8_NLCD_path, siteID=6, las_crs=2881):
    '''get elevation data around a given siteID'''

    las_list, L8_NLCD_tif = prep_file(las_path, L8_NLCD_path, siteID)

    # image shape
    x_max, y_max = 128, 128
    elevation = np.zeros((x_max, y_max))
    elevation_cls = elevation.copy()

    with rio.open(L8_NLCD_tif) as src:
        # for each pixel
        for x in range(x_max):
            for y in range(y_max):
                lon, lat = src.xy(x, y)
                z_avg = get_elevation(lat, lon, las_list, las_crs)
                elevation[x, y] = z_avg
                elevation_cls[x, y] = classify_z(z_avg)

    return elevation, elevation_cls

t_start = time.time()

elevation, elevation_cls = get_site_elevation(las_path, L8_NLCD_path, siteID=6, las_crs=2881)

t_end = time.time()
print ("Time elapsed: {} s".format(t_end - t_start))

elevation[:5,:5]

elevation_cls[:5,:5]
