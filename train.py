# -*- coding: utf-8 -*-

import os, sys, time, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split

def data_prep(patch_size=15):

    arr = np.load('./L8_NLCD_extracted_dataset.npy')

    # See the distri
    arr_nlcd = arr[:,8,:,:]
    freq = np.unique(arr_nlcd, return_counts=True)

    leg_num = np.array([0,11,12,21,22,23,24,31,41,42,43,51,52,71,72,73,74,81,82,90,95])
    leg_str = np.array(['NaN','Open Water','Perennial Ice/Snow','Developed, Open Space','Developed, Low Intensity',
            'Developed, Medium Intensity','Developed High Intensity','Barren Land (Rock/Sand/Clay)',
            'Deciduous Forest','Evergreen Forest','Mixed Forest','Dwarf Scrub','Shrub/Scrub',
            'Grassland/Herbaceous','Sedge/Herbaceous','Lichens','Moss','Pasture/Hay','Cultivated Crops',
            'Woody Wetlands','Emergent Herbaceous Wetlands'])

    # Conversion between original class and training labels
    nlcd_dic = {}

    # 9 classes, including NaN
    nlcd_dic['simple'] = {0:8,11:0,12:0,
                        21:1,22:1,23:1,24:1,31:2,41:3,42:3,
                        43:3,51:4,52:4,71:5,72:5,73:5,
                        74:5,81:6,82:6,90:7,95:7}

    nlcd_dic['developed'] = {21:0,22:1,23:2,24:3}

    sublist_labels = [21, 22, 23, 24]
    sublist_labels = [0,11,12,31,41,42,43,51,52,71,72,73,74,81,82,90,95]

    def get_patches(arr_l8, arr_nlcd, patch_size=5, mode='reduction', sublist=False):
        '''
        Get patches from numpy 3d or 4d array (multiband images)
        
        Band 8 stores labels 
        Each patch has patch_size*patch_size many pixels. Patch size is odd.
        '''
        s = patch_size
        patches = []
        targets = []
        if len(arr_l8.shape) == 4:
            '''
            Input has shape (N, X, Y, B): (sample_index, band, x-coor, y-coor)
            features has shape (M, patch_size, patch_size, B): (sample_index, band, x-coor, y-coor)
            '''
            N, X, Y, B = arr_l8.shape[0], arr_l8.shape[1], arr_l8.shape[2], arr_l8.shape[3]
            # Pixel (x, y) in sample n is the center of patches[m]
            # m= n*(X-s+1)*(Y-s+1) + (y-2)*(X-s+1) + (x-2), x,y,n starts from 0
            for n in range(N):
                for y in range(Y-s+1):         
                    for x in range(X-s+1):
                        if not sublist or arr_nlcd[n, x+s//2, y+s//2] in sublist_labels:
                            patches.append(arr_l8[n, x:x+s, y:y+s, :])
                            targets.append(nlcd_dic[mode][arr_nlcd[n, x+s//2, y+s//2]])

        if len(arr_l8.shape) == 3:
            '''
            Input has shape (B, X, Y): (band, x-coor, y-coor)
            features has shape (M, B, patch_size, patch_size): (sample_index, band, x-coor, y-coor)
            '''  
            X, Y, B = arr_l8.shape[0], arr_l8.shape[1], arr_l8.shape[2]
            # Pixel (x, y) is the center of patches[m], m=(y-2)*(X-s+1)+(x-2), x,y starts from 0
            for y in range(Y-s+1):         
                for x in range(X-s+1):
                    if not sublist or arr_nlcd[x+s//2, y+s//2] in sublist_labels:
                        patches.append(arr_l8[x:x+s, y:y+s, :])
                        targets.append(nlcd_dic[mode][arr_nlcd[x+s//2, y+s//2]])

        features = np.array(patches)
        targets = np.array(targets)
        return features, targets

    def map_nlcd(arr3d, mode='simple'):
        for x0 in range(arr3d.shape[0]):
            for x1 in range(arr3d.shape[1]):
                for x2 in range(arr3d.shape[2]):
                    arr3d[x0,x1,x2] = nlcd_dic[mode][arr3d[x0,x1,x2]]
        return arr3d            

    #########################################
    # Check outputs are well ordered
    # n = 3
    # x = 2
    # y = 4
    # s = 5
    # X = 128
    # Y = 128
    # m= n*(X-s+1)*(Y-s+1) + (y-2)*(X-s+1) + (x-2)

    # feat[m, :, 2, 2] - arr[n, :8, x, y]
    # arr[n, 8, x, y] - targ[m]



    arr_nlcd = arr[:,:,:,8].copy()
    arr_l8 = arr[:,:,:,:8].copy()

    prep = 'patch'

    if prep == 'patch':
        x, y = get_patches(arr_l8, arr_nlcd, 
                        patch_size=patch_size,
                        mode='simple',
                        sublist=False)
    else:
        x, y = arr_l8, map_nlcd(arr_nlcd)

    # Randomly split train/valid data using sklearn
    x, x_valid, y, y_valid = train_test_split(x, y, test_size=0.1, random_state=0)
    x_train = x
    y_train = y
    freq = np.unique(y_train, return_counts=True)

    num_train = x_train.shape[0]
    num_valid = x_valid.shape[0]

    return x_train, y_train, x_valid, y_valid

"""Training"""

from datetime import datetime
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, Dropout
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

def train(x_train, y_train, x_valid, y_valid ,patch_size=15, model_id = 'exp0.34'):

    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    '''CallBacks'''
    logdir = ('./' + model_id + '-log')
    for PATH in [logdir]:
        os.makedirs(PATH, exist_ok=True)

    from distutils.dir_util import copy_tree

    def get_callbacks(logdir):
        tb_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir,
                                                        update_freq='batch')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(logdir, model_id+'.hdf5'),
            save_weights_only=False,
            monitor='val_loss',
            mode='min',
            save_best_only=True)
        csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(logdir,'log.csv'), append=True)
        return [tb_callback, model_checkpoint_callback, csv_logger]
    callbacks = get_callbacks(logdir)

    n_classes = 9
    inputShape = (patch_size,patch_size, 8)

    y_train_categ = tf.keras.utils.to_categorical(y_train, num_classes=n_classes)
    y_valid_categ = tf.keras.utils.to_categorical(y_valid, num_classes=n_classes)

    inputShape = (patch_size, patch_size, 8)

    def patch_cnn_old():
        model = Sequential()
        model.add(Conv2D(8,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=inputShape))
        model.add(Conv2D(16,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(32,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(64,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Conv2D(128,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(3200, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        return model

    def patch_cnn():
        model = Sequential()
        model.add(tf.keras.layers.BatchNormalization(input_shape=inputShape))
        model.add(Conv2D(64,kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu',
                        activity_regularizer=l1))
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(64,kernel_size=(3, 3), strides=(1, 1), padding='same', 
                        activity_regularizer=l1, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(64,kernel_size=(3, 3), strides=(1, 1), padding='same', 
                        activity_regularizer=l1, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(64,kernel_size=(3, 3), strides=(1, 1), padding='same', 
                        activity_regularizer=l1, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Conv2D(32,kernel_size=(3, 3), strides=(1, 1), padding='same', 
                        activity_regularizer=l1, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Flatten())
        model.add(Dense(3200, activation='relu'))
        model.add(Dense(n_classes, activation='softmax'))
        return model

    with strategy.scope():
        dropout_rate = 0.1
        l1 = tf.keras.regularizers.l1(0)
        model = patch_cnn()
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                    optimizer=Adam(lr=0.0003),
                    metrics=['accuracy'] )

    model.summary()    
    epochs = 20
    hs = model.fit(x_train,y_train_categ, batch_size=1024,epochs=epochs, 
                    verbose=2,validation_data=(x_valid,y_valid_categ),
                    callbacks=callbacks,
                    )

if __name__ == '__main__':
    x_train, y_train, x_valid, y_valid = data_prep()
    train(x_train, y_train, x_valid, y_valid)