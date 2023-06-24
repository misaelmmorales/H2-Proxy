########################################################################################################################
###################################################### IMPORT PACKAGES #################################################
########################################################################################################################
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, LeakyReLU, Dropout1d, BatchNorm1d
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from torchviz import make_dot

class h2_cushion_rom(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes=[100]):
        self.slope = 0.33
        self.drop  = 0.1
        super(h2_cushion_rom, self).__init__()
        assert len(hidden_sizes) >= 1 , 'specify at least one hidden layer'
        layers = nn.ModuleList()
        layer_sizes = [in_features] + hidden_sizes
        for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.LeakyReLU(self.slope))
            layers.append(Dropout1d(self.drop))
            layers.append(BatchNorm1d(dim_out))
        self.layers = nn.Sequential(*layers)      
        self.out_layer = Linear(hidden_sizes[-1], out_features)
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.layers(out)
        out = self.out_layer(out)
        return out
    
class customLoss(nn.Module):
    def __init__(self):
        super(customLoss, self).__init__()
        self.l2_weight = 0.5
    def forward(self, pred, true):
        l1_loss = nn.MSELoss()(pred, true)
        l2_loss = nn.L1Loss()(pred, true)
        total_loss = (1-self.l2_weight)*l1_loss + self.l2_weight*l2_loss
        return total_loss
    
class H2Toolkit:
    def __init__(self):
        self.return_data = True
        self.verbose     = True
        self.save_result = True

        self.xcols       = range(12)
        self.ycols       = [12, 14, 15]
        self.noise_flag  = False
        self.noise       = [0, 0.05]
        self.gwrt_cutoff = 1e5
        self.std_outlier = 3

        self.valid_size  = 0.15
        self.test_size   = 0.20

    def check_torch_gpu(self):
        version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        print('Torch version: {}'.format(version))
        print('Torch build with CUDA? {}'.format(cuda_avail))
        print('# Device(s) available: {}, Name(s): {}'.format(count, name))
        self.device = torch.device('cuda' if cuda_avail else 'cpu')
        return None

    #################### PROCESSING ####################
    def read_data(self, n_subsample=None):
        data_ch4  = pd.read_csv('data_CH4.csv',  index_col=0)
        data_co2  = pd.read_csv('data_CO2.csv',  index_col=0)
        data_n2   = pd.read_csv('data_N2.csv',   index_col=0)
        data_nocg = pd.read_csv('data_NOCG.csv', index_col=0) 
        data_all  = pd.concat([data_ch4, data_co2, data_n2, data_nocg])
        if n_subsample:
            idx = np.random.randint(0, data_all.shape[0], n_subsample)
            self.all_data = data_all.iloc[idx,:]
        else:
            self.all_data = data_all
        if self.verbose:
            print('CH4: {} | CO2: {} | N2: {} | NOCG: {}'.format(data_ch4.shape, data_co2.shape, data_n2.shape, data_nocg.shape))
            print('All: {}'.format(self.all_data.shape))
        if self.return_data:
            return self.all_data

    def process_data(self, restype='SA or DGR'):
        if self.noise_flag:
            data = self.all_data + np.random.normal(self.noise[0], self.noise[1], (self.all_data.shape))
        else:
            data = self.all_data
        data_shuffle = data.sample(frac=1)
        data_trunc = data_shuffle[data_shuffle['gwrt']<self.gwrt_cutoff]
        data_outl = data_trunc[(np.abs(zscore(data_trunc))<self.std_outlier)]
        if restype=='SA':
            data_outl['Sw'] = 1
        elif restype=='DGR':
            data_outl['Sw'] = self.all_data['Sw']
        data_clean = data_outl.dropna()  
        X_data, y_data = data_clean.iloc[:, self.xcols], data_clean.iloc[:, self.ycols]
        y_data_log = y_data.copy()
        y_data_log['gwrt'] = np.log10(y_data['gwrt'])
        self.x_scaler, self.y_scaler = MinMaxScaler(), MinMaxScaler()
        self.x_scaler.fit(X_data);   self.y_scaler.fit(y_data_log)
        self.X_dataset = self.x_scaler.transform(X_data)
        self.y_dataset = self.y_scaler.transform(y_data_log)
        if self.verbose:
            print('Full dataset shape:', self.all_data.shape)
            print('Truncated dataset shape:', data_trunc.shape)
            print('Clean (no outliers) dataset shape:', data_clean.shape)
            print('Dataset: X={} | y={}'.format(self.X_dataset.shape, self.y_dataset.shape))
        if self.return_data:
            return self.X_dataset, self.y_dataset, self.x_scaler, self.y_scaler
        
    def train_valid_test_split(self):
        ntrain = int(np.floor(self.X_dataset.shape[0]*(1-self.test_size)))
        nvalid = ntrain - int(np.floor(ntrain * self.valid_size))
        self.X_train, self.y_train = self.X_dataset[:nvalid],       self.y_dataset[:nvalid]
        self.X_valid, self.y_valid = self.X_dataset[nvalid:ntrain], self.y_dataset[nvalid:ntrain]
        self.X_test,  self.y_test  = self.X_dataset[ntrain:],       self.y_dataset[ntrain:]
        if self.verbose:
            print('X: train: {} | validation: {} | test: {}'.format(self.X_train.shape, self.X_valid.shape, self.X_test.shape))
            print('y: train: {}  | validation: {}  | test: {}'.format(self.y_train.shape, self.y_valid.shape, self.y_test.shape))
        if self.return_data:
            return self.X_train, self.X_valid, self.X_test, self.y_train, self.y_valid, self.y_test
    
    def arrays_to_tensors(self):
        self.X_train_tensor = torch.tensor(self.X_train).float().to(self.device)
        self.y_train_tensor = torch.tensor(self.y_train).float().to(self.device)
        self.X_valid_tensor = torch.tensor(self.X_valid).float().to(self.device)
        self.y_valid_tensor = torch.tensor(self.y_valid).float().to(self.device)
        self.X_test_tensor = torch.tensor(self.X_test).float().to(self.device)
        self.y_test_tensor = torch.tensor(self.y_test).float().to(self.device)
        if self.return_data:
            return self.X_train_tensor, self.X_valid_tensor, self.X_test_tensor, self.y_train_tensor, self.y_valid_tensor, self.y_test_tensor
    
    def make_dataloader(self, train_shuffle=True, train_batch=40, valid_shuffle=False, valid_batch=10):
        self.train_dataset = TensorDataset(self.X_train_tensor, self.y_train_tensor)
        self.valid_dataset = TensorDataset(self.X_valid_tensor, self.y_valid_tensor)
        self.train_loader  = DataLoader(self.train_dataset, shuffle=train_shuffle, batch_size=train_batch)
        self.valid_loader  = DataLoader(self.valid_dataset, shuffle=valid_shuffle, batch_sampler=valid_batch)
        if self.return_data:
            return self.train_dataset, self.train_loader, self.valid_dataset, self.valid_loader
        
    #################### ROM #################### 
    def train_one_epoch(self, model, loss_func, optimizer):
        model.train()
        criterion = loss_func
        loss_step = []
        for (xbatch, ybatch) in self.train_loader:
            yhat = model(xbatch)
            loss = criterion(yhat, ybatch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_step.append(loss.item())
            train_loss_epoch = torch.tensor(loss_step).mean().numpy()
            return train_loss_epoch
    
    def validate(self, model, loss_func):
        model.eval()
        criterion = loss_func
        loss_step = []
        with torch.no_grad():
            for xbatch, ybatch in self.valid_loader:
                yhat = model(xbatch)
                val_loss = criterion(yhat, ybatch)
                loss_step.append(val_loss.item())
            val_loss_epoch = torch.tensor(loss_step).mean().numpy()
            return val_loss_epoch
        
    def train(self, model, loss_func, optimizer, epochs):
        model = model.to(self.device)
        dict_log = {'loss':[], 'valid_loss':[]}
        pbar = tqdm(range(epochs))
        for epoch in pbar:
            e = epoch/epochs
            loss_epoch = self.train_one_epoch(model, loss_func, optimizer)
            val_loss_epoch = self.validate(model, loss_func)
            msg = 'Epoch: {} -- Loss: {:.5f}, Validation Loss: {:.5f}'.format(e, loss_epoch, val_loss_epoch)
            pbar.set_description(msg)
            dict_log['loss'].append(loss_epoch)
            dict_log['valid_loss'].append(val_loss_epoch)
        if self.return_data:
            return dict_log



    def make_predictions(self):
        self.y_train_pred = None
        self.y_test_pred  = None
        if self.return_data:
            return self.y_train_pred, self.y_test_pred
    
    #################### PLOTS & PRINTS ####################
    def rescale_results(self):
        def inv_scale(data):
            return pd.DataFrame(self.y_scaler.inverse_transform(data), columns=data.columns)
        self.y_train_r,      self.y_test_r      = inv_scale(self.y_train),      inv_scale(self.y_test)
        self.y_train_pred_r, self.y_test_pred_r = inv_scale(self.y_train_pred), inv_scale(self.y_test_pred)
        self.y_train_r['gwrt'], self.y_train_pred_r['gwrt'] = 10**self.y_train_r['gwrt'], 10**self.y_train_pred_r['gwrt']
        self.y_test_r['gwrt'],  self.y_test_pred_r['gwrt']  = 10**self.y_test_r['gwrt'],  10**self.y_test_pred_r['gwrt']
        if self.return_data:
            return self.y_train_r, self.y_test_r, self.y_train_pred_r, self.y_test_pred_r
    
    def print_metrics(self):
        tot_train_r2,  tot_test_r2  = r2_score(self.y_train, self.y_train_pred),            r2_score(self.y_test, self.y_test_pred)
        tot_train_mse, tot_test_mse = mean_squared_error(self.y_train, self.y_train_pred),  mean_squared_error(self.y_test, self.y_test_pred)
        tot_train_mae, tot_test_mae = mean_absolute_error(self.y_train, self.y_train_pred), mean_absolute_error(self.y_test, self.y_test_pred)
        print('TRAIN: R2={:.3f} | MSE={:.5f} | MAE={:.5f}'.format(tot_train_r2, tot_train_mse, tot_train_mae))
        print('TEST:  R2={:.3f} | MSE={:.5f} | MAE={:.5f}'.format(tot_test_r2, tot_test_mse, tot_test_mae))
        for i in range(3):
            name = self.y_train.columns[i]
            trainr2,  testr2  = r2_score(self.y_train[name], self.y_train_pred[name]),            r2_score(self.y_test[name], self.y_test_pred[name])
            trainmse, testmse = mean_squared_error(self.y_train[name], self.y_train_pred[name]),  mean_squared_error(self.y_test[name], self.y_test_pred[name])
            trainmae, testmae = mean_absolute_error(self.y_train[name], self.y_train_pred[name]), mean_absolute_error(self.y_test[name], self.y_test_pred[name])
            print('\n')
            print('{} TRAIN: R2={:.3f}    | TEST: R2={:.3f}'.format(name, trainr2, testr2))
            print('{} TRAIN: MSE={:.5f} | TEST: MSE={:.5f}'.format(name, trainmse, testmse))   
            print('{} TRAIN: MAE={:.5f} | TEST: MAE={:.5f}'.format(name, trainmae, testmae))
        if self.save_data:
            metrics = np.array([tot_train_r2, tot_test_r2, tot_train_mse, tot_test_mse, tot_train_mae, tot_test_mae])
            np.save('metrics.npy', metrics)
            
    def plot_loss(self, title='', figsize=None):
        if figsize:
            plt.figure(figsize=figsize)
        loss, val = self.fit.history['loss'], self.fit.history['val_loss']
        epochs = len(loss)
        iterations = np.arange(epochs)
        plt.plot(iterations, loss, '-', label='loss')
        plt.plot(iterations, val,  '-', label='validation loss')
        plt.title(title+' Training Loss vs epochs'); plt.legend()
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.xticks(iterations[::epochs//10])
        plt.savefig('training_performance.png')
        plt.show()
       
    def plot_results(self, figsize=(15,4), figname='Results'):
        plt.figure(figsize=figsize)
        plt.suptitle(figname)
        for i in range(3):
            name = self.y_train.columns[i]
            plt.subplot(1,3,i+1)
            plt.axline([0,0],[1,1], c='r', linewidth=3)
            plt.scatter(self.y_train.iloc[:,i], self.y_train_pred.iloc[:,i], alpha=0.5, label='train')
            plt.scatter(self.y_test.iloc[:,i],  self.y_test_pred.iloc[:,i],  alpha=0.5, label='test')
            plt.xlabel('True'); plt.ylabel('Predicted'); plt.title('{}'.format(name)); plt.legend()
        plt.savefig(figname + '.png')
        plt.show()
        
    def save_data(self):
        if self.save_result:
            self.X_train.to_csv('X_train.csv'); self.X_test.to_csv('X_test.csv')
            self.y_train.to_csv('y_train.csv'); self.y_test.to_csv('y_test.csv')
            self.y_train_pred.to_csv('y_train_pred.csv'); self.y_test_pred.to_csv('y_test_pred.csv')          
            self.model.save('h2proxy_model', overwrite=True, save_format='h5')
            print('\nData is Saved! ..... DONE!')
        else:
            print('\n...DONE!')

########################################################################################################################
############################################################ END #######################################################
########################################################################################################################