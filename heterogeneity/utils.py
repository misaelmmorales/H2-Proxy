########################################################################################################################
###################################################### IMPORT PACKAGES #################################################
########################################################################################################################
import os, sys, glob, math, re, cv2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyvista as pv
from time import time

from scipy.stats import zscore
from scipy.io import loadmat, savemat
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.nn import Sequential
from torch.nn import Linear, ReLU, LeakyReLU, Dropout, BatchNorm1d
from torch.nn import Conv2d, ConvTranspose2d, MaxPool2d, BatchNorm2d
from torch.optim import NAdam
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot
import torchio as tio

class Heterogeneity:
    def __init__(self):
        self.return_data    = True
        self.ti_dir         = 'C:/Users/381792/Documents/MLTrainingImages/'
        #self.data_dir       = '//dcstorage.lanl.gov/MFR2/misael'
        self.data_dir       = '/project/MFR2/misael'
        self.n_realizations = 1000
        self.n_timesteps    = 61
        self.dim            = 256
        self.standard_dims  = (1000, 256,256)
        self.flat_dims      = (1000, 256*256)
        self.layers         = [48, 64, 80, 96]
        self.facies_range   = [0.75, 1.25]
        self.fluv_range     = [0.00, 2.80]
        self.gaus_range     = [0.00, 2.90]
        self.lognormnoise   = [-5.0, 0.1]
        self.theta          = [0, 30, 45, 60, 90, 120, 135, 150]
        self.seed           = 424242
        self.save_data      = False
        self.verbose        = True
        
    def check_torch_gpu(self):
        '''
        Check torch build in python to ensure GPU is available for training.
        '''
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        py_version, conda_env_name = sys.version, sys.executable.split('\\')[-2]
        print('-------------------------------------------------')
        print('------------------ VERSION INFO -----------------')
        print('Conda Environment: {} | Python version: {}'.format(conda_env_name, py_version))
        print('Torch version: {}'.format(torch_version))
        print('Torch build with CUDA? {}'.format(cuda_avail))
        print('# Device(s) available: {}, Name(s): {}\n'.format(count, name))
        self.device = torch.device('cuda' if cuda_avail else 'cpu')
        return None
    
    #################### DATA LOADING ####################    
    def make_facies(self):
        np.random.seed(self.seed)
        fname, k = {}, 0
        for root, dirs, files, in os.walk(self.ti_dir):
            for file in files:
                if file.endswith('.npy'):
                    fname[k] = os.path.join(self.ti_dir, root, file)
                    k += 1
        facies_raw = {}
        for i in range(len(fname)):
            facies_raw[i] = np.load(fname[i]).reshape(self.dim,self.dim,int(self.dim/2))[...,self.layers]
        f1 = np.array(list(facies_raw.values()))[...,0]
        f2 = np.array(list(facies_raw.values()))[...,1]
        f3 = np.array(list(facies_raw.values()))[...,2]
        f4 = np.array(list(facies_raw.values()))[...,3]
        idx = np.sort(np.random.choice(range(1272), self.n_realizations, replace=False))
        all_facies = np.concatenate([f1,f2,f3,f4])[idx]
        facies_norm = {}
        for i in range(self.n_realizations):
            scaler = MinMaxScaler((self.facies_range[0], self.facies_range[-1]))
            facies_norm[i] = scaler.fit_transform(all_facies[i].reshape(-1,1)).reshape(self.dim,self.dim)
        self.facies = np.array(list(facies_norm.values()))
        if self.save_data:
            if self.verbose:
                print('Saving Files...')
            pd.DataFrame(self.facies.reshape(self.flat_dims).T).iloc[:,:250].to_csv('data2D/facies_1_250.csv')
            pd.DataFrame(self.facies.reshape(self.flat_dims).T).iloc[:,250:500].to_csv('data2D/facies_250_500.csv')
            pd.DataFrame(self.facies.reshape(self.flat_dims).T).iloc[:,500:750].to_csv('data2D/facies_500_750.csv')
            pd.DataFrame(self.facies.reshape(self.flat_dims).T).iloc[:,750:].to_csv('data2D/facies_750_1000.csv')
        if self.verbose:
            print('...DONE!')
        if self.return_data:
            return self.facies

    def load_facies(self):
        f1 = pd.read_csv('data2D/facies_1_250.csv',    index_col=0)
        f2 = pd.read_csv('data2D/facies_250_500.csv',  index_col=0)
        f3 = pd.read_csv('data2D/facies_500_750.csv',  index_col=0)
        f4 = pd.read_csv('data2D/facies_750_1000.csv', index_col=0)
        self.facies = np.concatenate([f1,f2,f3,f4],-1).T.reshape(self.standard_dims)
        if self.verbose:
            print('Facies shape:', self.facies.shape)
        if self.return_data:
            return self.facies

    def load_perm_poro(self):
        np.random.seed(self.seed)
        perm_noAzim_1_125    = pd.read_csv('data2D/perm_noAzim_1_125.csv',    index_col=0)
        perm_noAzim_125_250  = pd.read_csv('data2D/perm_noAzim_125_250.csv',  index_col=0)
        perm_noAzim_250_375  = pd.read_csv('data2D/perm_noAzim_250_375.csv',  index_col=0)
        perm_noAzim_375_500  = pd.read_csv('data2D/perm_noAzim_375_500.csv',  index_col=0)
        perm_noAzim_500_625  = pd.read_csv('data2D/perm_noAzim_500_625.csv',  index_col=0)
        perm_noAzim_625_750  = pd.read_csv('data2D/perm_noAzim_625_750.csv',  index_col=0)
        perm_noAzim_750_875  = pd.read_csv('data2D/perm_noAzim_750_875.csv',  index_col=0)
        perm_noAzim_875_1000 = pd.read_csv('data2D/perm_noAzim_875_1000.csv', index_col=0)
        permf = np.hstack([perm_noAzim_1_125,   perm_noAzim_125_250, perm_noAzim_250_375, perm_noAzim_375_500,
                        perm_noAzim_500_625, perm_noAzim_625_750, perm_noAzim_750_875, perm_noAzim_875_1000]).T
        perm_0azim   = pd.read_csv('data2D/perm_0azim.csv')
        perm_30azim  = pd.read_csv('data2D/perm_30azim.csv')
        perm_45azim  = pd.read_csv('data2D/perm_45azim.csv')
        perm_60azim  = pd.read_csv('data2D/perm_60azim.csv')
        perm_90azim  = pd.read_csv('data2D/perm_90azim.csv')
        perm_120azim = pd.read_csv('data2D/perm_120azim.csv')
        perm_135azim = pd.read_csv('data2D/perm_135azim.csv')
        perm_150azim = pd.read_csv('data2D/perm_150azim.csv')
        permg = np.hstack([perm_0azim,   perm_30azim,  perm_45azim,  perm_60azim, 
                        perm_90azim,  perm_120azim, perm_135azim, perm_150azim]).T
        if self.verbose:
            print('Permf: {} | Permg: {}'.format(permf.shape, permg.shape))
        # Fluvial fields
        temp_fluv = permf.reshape(self.flat_dims).T
        temp_fluv_norm = MinMaxScaler((self.fluv_range[0], self.fluv_range[-1])).fit_transform(temp_fluv).T.reshape(self.standard_dims)
        perm_fluv = temp_fluv_norm * self.facies
        poro_fluv = 10**((perm_fluv-7)/10)
        # Gaussian fields
        temp_gaus = permg.reshape(self.flat_dims).T
        temp_gaus_norm = MinMaxScaler((self.gaus_range[0], self.gaus_range[-1])).fit_transform(temp_gaus).T.reshape(self.standard_dims)
        perm_gaus = temp_gaus_norm + np.random.lognormal(self.lognormnoise[0], self.lognormnoise[1], (self.standard_dims))
        poro_gaus = 10**((perm_gaus-7)/10)
        # Heterogeneity (perm, poro)
        self.hete_fluv = np.concatenate([np.expand_dims(perm_fluv,-1), np.expand_dims(poro_fluv,-1)], -1)
        self.hete_gaus = np.concatenate([np.expand_dims(perm_gaus,-1), np.expand_dims(poro_gaus,-1)], -1)
        if self.verbose:
            print('Fluvial heterogeneity: {} | Gaussian heterogeneity: {}'.format(self.hete_fluv.shape, self.hete_gaus.shape))
        if self.return_data:
            return self.hete_fluv, self.hete_gaus        
        
    def load_datasets(self):
        hete_fluv = np.zeros((self.n_realizations,self.dim,self.dim,2))
        hete_gaus = np.zeros((self.n_realizations,self.dim,self.dim,2))
        facies = np.zeros(self.standard_dims)
        for i in range(self.n_realizations):
            hete_fluv[i] = np.load('data2D/hete_fluv/fluvial{}.npy'.format(i))
            hete_gaus[i] = np.load('data2D/hete_gaus/gaussian{}.npy'.format(i))
            facies[i] = np.array(pd.read_csv('data2D/facies/facies{}.csv'.format(i), index_col=0))
        mask_0azim   = np.zeros((125,self.dim,self.dim))
        mask_30azim  = np.zeros((125,self.dim,self.dim))+30
        mask_45azim  = np.zeros((125,self.dim,self.dim))+45
        mask_60azim  = np.zeros((125,self.dim,self.dim))+60
        mask_90azim  = np.zeros((125,self.dim,self.dim))+90
        mask_120azim = np.zeros((125,self.dim,self.dim))+120
        mask_135azim = np.zeros((125,self.dim,self.dim))+135
        mask_150azim = np.zeros((125,self.dim,self.dim))+150
        angle_mask = np.concatenate([mask_0azim, mask_30azim, mask_45azim, mask_60azim, mask_90azim, mask_120azim, mask_135azim, mask_150azim])
        self.fluvial_dataset = np.concatenate([hete_fluv, np.expand_dims(facies,-1)], -1)
        self.gaussian_dataset = np.concatenate([hete_gaus, np.expand_dims(angle_mask,-1)], -1)
        if self.verbose:
            print('Fluvial: {} | Gaussian: {}'.format(self.fluvial_dataset.shape, self.gaussian_dataset.shape))    
        if self.return_data:
            return self.fluvial_dataset, self.gaussian_dataset        
        
    # def load_perm_poro_3d(self):
    #     perm_0azim   = pd.read_csv('data3D/perm_0azim.csv')
    #     perm_30azim  = pd.read_csv('data3D/perm_30azim.csv')
    #     perm_45azim  = pd.read_csv('data3D/perm_45azim.csv')
    #     perm_60azim  = pd.read_csv('data3D/perm_60azim.csv')
    #     perm_90azim  = pd.read_csv('data3D/perm_90azim.csv')
    #     perm_120azim = pd.read_csv('data3D/perm_120azim.csv')
    #     perm_135azim = pd.read_csv('data3D/perm_135azim.csv')
    #     perm_150azim = pd.read_csv('data3D/perm_150azim.csv')
    #     perm_noAzim_1 = pd.read_csv('data3D/perm_noazim_3d_1_65.csv')
    #     perm_noAzim_2 = pd.read_csv('data3D/perm_noazim_3d_65_130.csv')
    #     perm_noAzim_3 = pd.read_csv('data3D/perm_noazim_3d_130_195.csv')
    #     perm_noAzim_4 = pd.read_csv('data3D/perm_noazim_3d_195_260.csv')
    #     perm_noAzim_5 = pd.read_csv('data3D/perm_noazim_3d_260_325.csv')
    #     perm_noAzim_6 = pd.read_csv('data3D/perm_noazim_3d_325_390.csv')
    #     perm_noAzim_7 = pd.read_csv('data3D/perm_noazim_3d_390_455.csv')
    #     perm_noAzim_8 = pd.read_csv('data3D/perm_noazim_3d_455_520.csv')
    #     permg = np.hstack([perm_0azim, perm_30azim, perm_45azim, perm_60azim, 
    #                        perm_90azim, perm_120azim, perm_135azim, perm_150azim]).T
    #     permf = np.hstack([perm_noAzim_1, perm_noAzim_2, perm_noAzim_3, perm_noAzim_4,
    #                        perm_noAzim_5, perm_noAzim_6, perm_noAzim_7, perm_noAzim_8]).T
    #     permf_norm = MinMaxScaler((0, 2.80)).fit_transform(permf.T).T
    #     permg_norm = MinMaxScaler((0, 2.90)).fit_transform(permg.T).T
    #     facies = np.zeros((520,128*128*16))
    #     for i in range(520):
    #         facies[i] = np.load('data3D/facies/facies{}.npy'.format(i)).reshape(128*128*16)
    #     perm_fluv = permf_norm * facies
    #     poro_fluv = 10**((perm_fluv-7)/10)
    #     k = np.expand_dims(perm_fluv.reshape(520,128,128,16), -1)
    #     p = np.expand_dims(poro_fluv.reshape(520,128,128,16), -1)
    #     hete_fluv = np.concatenate([k,p], -1)
    #     return Nones
    
def double_convolution(in_channels, out_channels):
    conv_op = Sequential(
        Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ReLU(inplace=True),
        Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ReLU(inplace=True))
    return conv_op

class h2_hete_rom(nn.Module):
    def __init__(self):
        super(h2_hete_rom, self).__init__()
        
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_convolution_1 = double_convolution(4, 16)
        self.down_convolution_2 = double_convolution(16, 32)
        self.down_convolution_3 = double_convolution(32, 64)
        self.down_convolution_4 = double_convolution(64, 128)
        self.down_convolution_5 = double_convolution(128, 256)

        self.up_transpose_1   = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_convolution_1 = double_convolution(256, 128)
        self.up_transpose_2   = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_convolution_2 = double_convolution(128, 64)
        self.up_transpose_3   = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.up_convolution_3 = double_convolution(64, 32)
        self.up_transpose_4   = nn.ConvTranspose2d(32, 16, kernel_size=2,  stride=2)
        self.up_convolution_4 = double_convolution(32, 16)

        self.out = nn.Conv2d(16, 2, kernel_size=1) 
        
    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)        
        
        up_1 = self.up_transpose_1(down_9)
        x    = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x    = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x    = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x    = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        out   = self.out(x)
        return out