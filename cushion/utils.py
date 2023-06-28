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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, LeakyReLU, Dropout1d, Dropout, BatchNorm1d
from torch.nn.init import kaiming_uniform_, constant_
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
from torchviz import make_dot

class h2_cushion_rom(nn.Module):
    def __init__(self, in_features, out_features, hidden_sizes=[100,60,10], 
                 act=nn.LeakyReLU(0.3), drop=nn.Dropout(0.1)):
        super(h2_cushion_rom, self).__init__()
        assert len(hidden_sizes) >= 1 , 'specify at least one hidden layer'
        layers = nn.ModuleList()
        layer_sizes = [in_features] + hidden_sizes
        for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(act)
            layers.append(drop)
            layers.append(BatchNorm1d(dim_out))
        self.layers = nn.Sequential(*layers)
        self.out_layer = Linear(hidden_sizes[-1], out_features)
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.layers(out)
        out = self.out_layer(out)
        return out
    
class L1L2_Loss(nn.Module):
    def __init__(self):
        super(L1L2_Loss, self).__init__()
        self.l2_weight = 0.5
    def forward(self, pred, true):
        l1_loss = nn.MSELoss()(pred, true)
        l2_loss = nn.L1Loss()(pred, true)
        total_loss = (1-self.l2_weight)*l1_loss + self.l2_weight*l2_loss
        return total_loss
    
class H2Toolkit:
    def __init__(self):
        self.return_data = False
        self.return_plot = True
        self.save_data   = True
        self.verbose     = True
        self.inp         = 12
        self.out         = 3
        self.xcols       = range(12)
        self.ycols       = [12, 14, 15]
        self.noise_flag  = False
        self.noise       = [0, 0.05]
        self.gwrt_cutoff = 1e5
        self.std_outlier = 3
        self.y_labels    = ['efft','ymft','gwrt']

        self.test_size   = 0.25
        self.epochs      = 250
        self.batch_size  = 100
        self.monitor_epochs = 50

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
        data_ch4  = pd.read_csv('data/data_CH4.csv',  index_col=0)
        data_co2  = pd.read_csv('data/data_CO2.csv',  index_col=0)
        data_n2   = pd.read_csv('data/data_N2.csv',   index_col=0)
        data_nocg = pd.read_csv('data/data_NOCG.csv', index_col=0) 
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
            data_outl.loc[:,'Sw'] = 1
        self.data_clean = data_outl.dropna()  
        X_data, y_data = self.data_clean.iloc[:, self.xcols], self.data_clean.iloc[:, self.ycols]
        y_data_log = pd.DataFrame(columns=y_data.columns)
        y_data_log['efft'], y_data_log['ymft'] = y_data['efft'], y_data['ymft']
        y_data_log['gwrt'] = np.log10(y_data['gwrt'])
        self.x_scaler, self.y_scaler = MinMaxScaler(), MinMaxScaler()
        self.x_scaler.fit(X_data);   self.y_scaler.fit(y_data_log)
        self.X_dataset = self.x_scaler.transform(X_data)
        self.y_dataset = self.y_scaler.transform(y_data_log)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_dataset, self.y_dataset, test_size=self.test_size)
        if self.verbose:
            print('Full dataset shape:', self.all_data.shape)
            print('Truncated dataset shape:', data_trunc.shape)
            print('Clean (no outliers) dataset shape:', self.data_clean.shape)
            print('Dataset: X={} | y={}'.format(self.X_dataset.shape, self.y_dataset.shape))
            print('X_train: {} | y_train: {}\nX_test:  {} | y_test:  {}'.format(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape))
        if self.return_data:
            datasets = {'X_dataset':self.X_dataset, 'y_dataset':self.y_dataset}
            train_test_data = {'X_train':self.X_train, 'X_test':self.X_test, 'y_train':self.y_train, 'y_test':self.y_test}
            scalers = {'x_scaler':self.x_scaler, 'y_scaler':self.y_scaler}
            return datasets, train_test_data, scalers 
        
    def load_data(self):
        self.X_dataset = np.load('data/X_data.npy')
        self.y_dataset = np.load('data/y_data.npy')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_dataset, self.y_dataset, test_size=self.test_size)
        if self.verbose:
            print('Dataset: X={} | y={}'.format(self.X_dataset.shape, self.y_dataset.shape))
            print('X_train: {} | y_train: {}\nX_test:  {} | y_test:  {}'.format(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape))
        if self.return_data:
            datasets = {'X_dataset':self.X_dataset, 'y_dataset':self.y_dataset}
            train_test_data = {'X_train':self.X_train, 'X_test':self.X_test, 'y_train':self.y_train, 'y_test':self.y_test}
            return datasets, train_test_data 
            
    #################### ROM #################### 
    def train(self, model, loss_fn, optimizer, validation_split=0.2):
        #tensorize
        self.X_train_tensor = torch.Tensor(self.X_train).to(self.device)
        self.X_test_tensor  = torch.Tensor(self.X_test).to(self.device)
        self.y_train_tensor = torch.Tensor(self.y_train).to(self.device)
        # Training loop
        model = model.to(self.device)
        loss, validation_loss = [], []
        self.fit = {'loss':[], 'validation_loss':[]}
        start = time()
        for epoch in range(self.epochs):
            xtrain, xvalid, ytrain, yvalid = train_test_split(self.X_train_tensor, self.y_train_tensor, test_size=validation_split)
            # training
            model.train()
            epoch_loss = 0.0
            for i in range(0, len(xtrain), self.batch_size):
                inp  = torch.Tensor(xtrain[i:i+self.batch_size]).to(self.device)
                true = torch.Tensor(ytrain[i:i+self.batch_size]).to(self.device)
                optimizer.zero_grad()
                pred = model(inp)
                loss = loss_fn(pred, true)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()*inp.size(0)
            self.fit['loss'].append(epoch_loss/len(xtrain))
            # validation
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(xvalid), self.batch_size):
                    inp = torch.Tensor(xvalid[i:i+self.batch_size]).to(self.device)
                    true = torch.Tensor(yvalid[i:i+self.batch_size]).to(self.device)
                    pred = model(inp)
                    validation_loss += loss_fn(pred, true).item()*inp.size(0)
            self.fit['validation_loss'].append(validation_loss/len(xvalid))
            if epoch%self.monitor_epochs==0:
                print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, self.epochs, epoch_loss/len(xtrain), validation_loss/len(xvalid)))
        traintime = (time() - start)/60
        n_params  = sum(p.numel() for p in model.parameters())
        if self.verbose:
            print('# Parameters: {} | Training time: {:.3f} minutes'.format(n_params, traintime))
        if self.save_data:
            torch.save(model.state_dict(), 'h2_cushion_rom.pt')            
            
    def make_predictions(self, model):
        model = model.to(self.device)
        self.y_train_pred = np.array(model(self.X_train_tensor).tolist())
        self.y_test_pred  = np.array(model(self.X_test_tensor).tolist())
        return self.y_train_pred, self.y_test_pred
    
    #################### PLOTS & PRINTS ####################
    def print_metrics(self):
        self.tot_train_r2,  self.tot_test_r2  = r2_score(self.y_train, self.y_train_pred),  r2_score(self.y_test, self.y_test_pred)
        tot_train_mse, tot_test_mse = mean_squared_error(self.y_train, self.y_train_pred),  mean_squared_error(self.y_test, self.y_test_pred)
        tot_train_mae, tot_test_mae = mean_absolute_error(self.y_train, self.y_train_pred), mean_absolute_error(self.y_test, self.y_test_pred)
        print('TRAIN: R2={:.3f} | MSE={:.5f} | MAE={:.5f}'.format(self.tot_train_r2, tot_train_mse, tot_train_mae))
        print('TEST:  R2={:.3f} | MSE={:.5f} | MAE={:.5f}'.format(self.tot_test_r2, tot_test_mse, tot_test_mae))     
        for i in range(3):
            name = self.y_labels[i]
            trainr2,  testr2  = r2_score(self.y_train[:,i], self.y_train_pred[:,i]),            r2_score(self.y_test[:,i], self.y_test_pred[:,i])
            trainmse, testmse = mean_squared_error(self.y_train[:,i], self.y_train_pred[:,i]),  mean_squared_error(self.y_test[:,i], self.y_test_pred[:,i])
            trainmae, testmae = mean_absolute_error(self.y_train[:,i], self.y_train_pred[:,i]), mean_absolute_error(self.y_test[:,i], self.y_test_pred[:,i])
            print('---------------')
            print('{} TRAIN: R2={:.3f}    | TEST: R2={:.3f}'.format(name, trainr2, testr2))
            print('{} TRAIN: MSE={:.5f} | TEST: MSE={:.5f}'.format(name, trainmse, testmse))   
            print('{} TRAIN: MAE={:.5f} | TEST: MAE={:.5f}'.format(name, trainmae, testmae))
        if self.save_data:
            metrics = np.array([self.tot_train_r2, self.tot_test_r2, tot_train_mse, tot_test_mse, tot_train_mae, tot_test_mae])
            np.save('data/metrics.npy', metrics)
            
    def plot_loss(self, title='', figsize=(4,3)):
        if figsize:
            plt.figure(figsize=figsize)
        loss, val = self.fit['loss'], self.fit['validation_loss']
        epochs = len(loss)
        iterations = np.arange(epochs)
        plt.plot(iterations, loss, '-', label='loss')
        plt.plot(iterations, val,  '-', label='validation loss')
        plt.title(title+' Training Loss vs epochs'); plt.legend()
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.xticks(iterations[::epochs//10])
        plt.savefig('figures/training_performance.png')
        if self.return_plot:
            plt.show()
       
    def plot_results(self, figsize=(15,4), figname='Results'):
        plt.figure(figsize=figsize)
        plt.suptitle(figname + ' -- Train $R^2$={:.2f} | Test $R^2$={:.2f}'.format(self.tot_train_r2, self.tot_test_r2))
        for i in range(3):
            name = self.y_labels[i]
            r2train = r2_score(self.y_train[:,i], self.y_train_pred[:,i])
            r2test  = r2_score(self.y_test[:,i],  self.y_test_pred[:,i])
            plt.subplot(1,3,i+1)
            plt.scatter(self.y_train[:,i], self.y_train_pred[:,i], alpha=0.5, label='train')
            plt.scatter(self.y_test[:,i],  self.y_test_pred[:,i],  alpha=0.5, label='test')
            plt.axline([0,0],[1,1], c='r', linewidth=3); plt.legend()
            plt.xlabel('True'); plt.ylabel('Predicted'); plt.xlim([-0.1,1.1]); plt.ylim([-0.1,1.1])
            plt.title('{} - $R^2_{}$={:.2f} ; $R^2_{}$={:.2f}'.format(name,'{train}',r2train,'{test}',r2test))
        plt.savefig('figures/' + figname + '.png')
        if self.return_plot:
            plt.show()
            
########################################################################################################################
############################################################ END #######################################################