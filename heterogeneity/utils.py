########################################################################################################################
###################################################### IMPORT PACKAGES #################################################
########################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from time import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import keras
from keras.backend import clear_session
import tensorflow as tf

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Checking Tensorflow Version:')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
        
class H2_heterogeneity:
    def __init__(self):
        self.verbose      = False
        self.dim          = 256
        self.layer        = 64
        self.ti_dir       = 'C:/Users/381792/Documents/MLTrainingImages/'
        self.selector     = [2, 33, 95, 140, 174, 184, 198, 236, 304]
        self.angles       = [0, 30, 45, 60, 90, 120, 135, 150, 180]
        self.scaler_range = [0.75,1.25]
        self.standard_dim = (9,256,256)
        self.return_data  = False
        self.save_data    = False
        
    def create_data(self):
        fname, self.fluv_name, facies_raw, facies_norm = {}, {}, {}, {}
        k = 0
        for root, dirs, files in os.walk(self.ti_dir):
            for file in files:
                if file.endswith('.npy'):
                    fname[k] = os.path.join(self.ti_dir,root,file)
                    k += 1                
        for i  in range(len(self.selector)):
            if i >= len(self.selector)-2:
                self.fluv_name[i] = fname[self.selector[i]][43:].split('\\')[:1]
            else:
                self.fluv_name[i] = fname[self.selector[i]][43:].split('\\')[:2]
        for i in range(len(self.selector)):
            facies_raw[i] = np.load(fname[self.selector[i]]).reshape(self.dim,self.dim,128)[...,self.layer]
        for i in range(len(self.selector)):
            scaler = MinMaxScaler((self.scaler_range[0],self.scaler_range[1]))
            facies_norm[i] = scaler.fit_transform(facies_raw[i].reshape(-1,1)).reshape(self.dim,self.dim)
        self.facies = np.array(list(facies_norm.values()))
        self.fluvial_perm_raw = np.array(pd.read_csv('perm2d_NoAzim.csv').T).reshape(self.standard_dim)
        self.gaussian_perm_raw = np.array(pd.read_csv('perm2d_Azim.csv').T).reshape(self.standard_dim)
        if self.verbose:
            print('Facies Realizations shape:', self.facies.shape)
            print('Fluvial Realizations shape:', self.fluvial_perm_raw.shape)
            print('Gaussian Realizations shape:', self.gaussian_perm_raw.shape)
        if self.return_data:
            return self.facies, self.fluvial_perm_raw, self.gaussian_perm_raw
            
    def process_perm_poro(self, lognormnoise=[-5,.1]):
        # Fluvial fields
        temp_fluv = self.fluvial_perm_raw.reshape(len(self.selector), self.dim*self.dim).T
        temp_fluv_norm = MinMaxScaler((0,2.9)).fit_transform(temp_fluv).T.reshape(self.standard_dim)
        self.perm_fluv = temp_fluv_norm * self.facies
        self.poro_fluv = 10**((self.perm_fluv-7)/10)
        # Gaussian fields
        temp_gaus = self.gaussian_perm_raw.reshape(len(self.selector), self.dim*self.dim).T
        temp_gaus_norm = MinMaxScaler((0,3.4)).fit_transform(temp_gaus).T.reshape(self.standard_dim)
        self.perm_gaus = temp_gaus_norm + np.random.lognormal(lognormnoise[0], lognormnoise[1], (self.standard_dim))
        self.poro_gaus = 10**((self.perm_gaus-7)/10)
        # concatenate into a single array
        self.df = np.concatenate([np.expand_dims(self.perm_fluv,-1), np.expand_dims(self.poro_fluv,-1), 
                                  np.expand_dims(self.perm_gaus,-1), np.expand_dims(self.poro_gaus,-1)], -1)
        if self.verbose:
            print('Dataset shape:', self.df.shape)
        if self.save_data:
            np.save('heterogeneity.npy', self.df)
        if self.return_data:
            return self.df
        
    def plot_samples(self, figsize=(20,6), cmaps=['jet','jet']):
        fig, axs = plt.subplots(4, len(self.selector), figsize=figsize)
        im, k = {}, 0
        for i in range(4):
            for j in range(len(self.selector)):
                axs[0,j].set(title='Realization {}'.format(j+1))
                axs[i,j].set(xticks=[], yticks=[])
                if i%2==0:
                    im[k] = axs[i,j].imshow(self.df[j,:,:,i], cmap=cmaps[0])
                    axs[i,0].set(ylabel='Log-Perm')
                else:
                    im[k] = axs[i,j].imshow(self.df[j,:,:,i], cmap=cmaps[1])
                    axs[i,0].set(ylabel='Porosity')
                k += 1
        for k in range(4*len(self.selector)):
            plt.colorbar(im[k], fraction=0.046, pad=0.04)
        plt.show()