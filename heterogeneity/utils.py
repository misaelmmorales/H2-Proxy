########################################################################################################################
###################################################### IMPORT PACKAGES #################################################
########################################################################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class Heterogeneity:
    def __init__(self):
        self.return_data    = True
        self.ti_dir         = 'C:/Users/381792/Documents/MLTrainingImages/'
        self.n_realizations = 1000
        self.dim            = 256
        self.standard_dims  = (1000,256,256)
        self.flat_dims      = (1000, 256*256)
        self.layers         = [48,64,80,96]
        self.facies_range   = [0.75, 1.25]
        self.fluv_range     = [0, 2.25]
        self.gaus_range     = [0, 2.90]
        self.lognormnoise   = [-5, 0.1]
        self.seed           = 424242
        self.save_data      = False
        self.verbose        = True
        
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
            pd.DataFrame(self.facies.reshape(self.flat_dims).T).iloc[:,:250].to_csv('data/facies_1_250.csv')
            pd.DataFrame(self.facies.reshape(self.flat_dims).T).iloc[:,250:500].to_csv('data/facies_250_500.csv')
            pd.DataFrame(self.facies.reshape(self.flat_dims).T).iloc[:,500:750].to_csv('data/facies_500_750.csv')
            pd.DataFrame(self.facies.reshape(self.flat_dims).T).iloc[:,750:].to_csv('data/facies_750_1000.csv')
        if self.verbose:
            print('...DONE!')
        if self.return_data:
            return self.facies

    def load_facies(self):
        f1 = pd.read_csv('data/facies_1_250.csv',    index_col=0)
        f2 = pd.read_csv('data/facies_250_500.csv',  index_col=0)
        f3 = pd.read_csv('data/facies_500_750.csv',  index_col=0)
        f4 = pd.read_csv('data/facies_750_1000.csv', index_col=0)
        self.facies = np.concatenate([f1,f2,f3,f4],-1).T.reshape(self.standard_dims)
        if self.verbose:
            print('Facies shape:', self.facies.shape)
        if self.return_data:
            return self.facies

    def load_perm_poro(self):
        np.random.seed(self.seed)
        perm_noAzim_1_125    = pd.read_csv('data/perm_noAzim_1_125.csv',    index_col=0)
        perm_noAzim_125_250  = pd.read_csv('data/perm_noAzim_125_250.csv',  index_col=0)
        perm_noAzim_250_375  = pd.read_csv('data/perm_noAzim_250_375.csv',  index_col=0)
        perm_noAzim_375_500  = pd.read_csv('data/perm_noAzim_375_500.csv',  index_col=0)
        perm_noAzim_500_625  = pd.read_csv('data/perm_noAzim_500_625.csv',  index_col=0)
        perm_noAzim_625_750  = pd.read_csv('data/perm_noAzim_625_750.csv',  index_col=0)
        perm_noAzim_750_875  = pd.read_csv('data/perm_noAzim_750_875.csv',  index_col=0)
        perm_noAzim_875_1000 = pd.read_csv('data/perm_noAzim_875_1000.csv', index_col=0)
        permf = np.hstack([perm_noAzim_1_125,   perm_noAzim_125_250, perm_noAzim_250_375, perm_noAzim_375_500,
                        perm_noAzim_500_625, perm_noAzim_625_750, perm_noAzim_750_875, perm_noAzim_875_1000]).T
        perm_0azim   = pd.read_csv('data/perm_0azim.csv')
        perm_30azim  = pd.read_csv('data/perm_30azim.csv')
        perm_45azim  = pd.read_csv('data/perm_45azim.csv')
        perm_60azim  = pd.read_csv('data/perm_60azim.csv')
        perm_90azim  = pd.read_csv('data/perm_90azim.csv')
        perm_120azim = pd.read_csv('data/perm_120azim.csv')
        perm_135azim = pd.read_csv('data/perm_135azim.csv')
        perm_150azim = pd.read_csv('data/perm_150azim.csv')
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