########################################################################################################################
###################################################### IMPORT PACKAGES #################################################
########################################################################################################################
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, LeakyReLU, Dropout, BatchNorm1d
from torch.optim import NAdam
import torch.nn.functional as F
from torchviz import make_dot

class h2_cushion_rom(nn.Module):
    '''
    H2 Cushion Gas proxy model: a torch dense nn to estimate [efft,ymft,gwrt] from various 
    geologic and operational parameters.
    '''
    def __init__(self, in_features, out_features, hidden_sizes=[100,60,10]):
        super(h2_cushion_rom, self).__init__()
        assert len(hidden_sizes) >= 1 , 'specify at least one hidden layer'
        layers = nn.ModuleList()
        layer_sizes = [in_features] + hidden_sizes
        for dim_in, dim_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(Linear(dim_in, dim_out))              #fully-connected
            layers.append(Dropout(0.125))                       #dropout
            layers.append(BatchNorm1d(dim_out))                 #batch normalization
            layers.append(LeakyReLU(0.2))                       #activation
        self.layers = nn.Sequential(*layers)                    #serialize
        self.out_layer = Linear(hidden_sizes[-1], out_features) #output layer
    def forward(self, x):
        out = x.view(x.shape[0], -1)
        out = self.layers(out)
        out = self.out_layer(out)
        return out

class Custom_Loss(nn.Module):
    '''
    custom loss function with weighted average combination of (smooth) L1 and L2 metrics.
    '''
    def __init__(self):
        super(Custom_Loss, self).__init__()
        self.l2_weight = 0.1
    def forward(self, pred, true):
        l1_loss = nn.MSELoss()(pred, true)          #l2 loss
        l2_loss = nn.SmoothL1Loss()(pred, true)     #l1 loss (smooth - Huber/L1)
        total_loss = (1-self.l2_weight)*l1_loss + self.l2_weight*l2_loss
        return total_loss

class H2Toolkit:
    '''
    a large module for self-contained for:
    (1) data loading and processing,
    (2) defining the proxy model, training, and predictions,
    (3) computing performance metrics and plotting results.
    '''
    def __init__(self):
        self.return_data = False                    #return data?
        self.return_plot = False                    #print plots?
        self.save_data   = True                     #save data?
        self.verbose     = True                     #print outputs?
        self.inp         = 12                       #n features
        self.out         = 3                        #n targets
        self.xcols       = range(12)                #feature columns
        self.ycols       = [12, 14, 15]             #target columns
        self.noise_flag  = False                    #add noise?
        self.noise       = [0, 0.05]                #added noise mean, std
        self.gwrt_cutoff = 1e5                      #GWRT outlier threshold
        self.std_outlier = 3                        #target outlier threshold
        self.y_labels    = ['efft','ymft','gwrt']   #target names
        self.test_size   = 0.25                     #train-(test) split size
        self.epochs      = 100                      #training epochs
        self.batch_size  = 256                      #batch size
        self.delta_epoch = 10                       #print performance every n epochs

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
    
    #################### ROM ####################
    def train(self, validation_split=0.2):
        '''
        Call torch model, set optimizer & loss, and train .
        '''
        # model parameters
        self.model = h2_cushion_rom(self.inp, self.out, hidden_sizes=[512,256,128,64]).to(self.device)
        optimizer = NAdam(self.model.parameters(), lr=2e-3)
        loss_fn = Custom_Loss()
        # tensorize
        self.X_train_tensor = torch.Tensor(self.X_train).to(self.device) #tensorize X_train to gpu
        self.y_train_tensor = torch.Tensor(self.y_train).to(self.device) #tensorize y_train to gpu
        # Training loop
        loss, validation_loss = [], []                                   #initialize losses to 0
        self.metrics = {'loss':[], 'validation_loss':[]}                 #define train/validation metrics dict
        start = time()                                                   #start training loop timer
        xtrain, xvalid, ytrain, yvalid = train_test_split(self.X_train_tensor, self.y_train_tensor, test_size=validation_split)
        print('----------------- MODEL TRAINING ----------------')
        for epoch in range(self.epochs):
            # training
            self.model.train()                                                   #set model as trainable
            epoch_loss = 0.0                                                     #current epoch loss=0
            for i in range(0, len(xtrain), self.batch_size):
                inp  = torch.Tensor(xtrain[i:i+self.batch_size]).to(self.device) #tensorize xbatch to gpu
                true = torch.Tensor(ytrain[i:i+self.batch_size]).to(self.device) #tensorize ybatch to gpu
                optimizer.zero_grad()                                            #reset opt gradients
                pred = self.model(inp)                                           #predict y_=f(xbatch)
                loss = loss_fn(pred, true)                                       #define loss
                loss.backward()                                                  #backpropagation
                optimizer.step()                                                 #optimizer iteration
                epoch_loss += loss.item()*inp.size(0)                            #current epoch loss
            self.metrics['loss'].append(epoch_loss/len(xtrain))
            # validation
            self.model.eval()                                                    #set model as non-trainable
            validation_loss = 0.0                                                #current epoch val_loss=0
            with torch.no_grad():                                                #non-trainable
                for i in range(0, len(xvalid), self.batch_size):
                    inp  = torch.Tensor(xvalid[i:i+self.batch_size]).to(self.device) #tensorize xvbatch to gpu
                    true = torch.Tensor(yvalid[i:i+self.batch_size]).to(self.device) #tensorize yvbatch to gpu
                    pred = self.model(inp)                                       #predict yv_=f(yvbatch)
                    validation_loss += loss_fn(pred, true).item()*inp.size(0)    #define loss and update
            self.metrics['validation_loss'].append(validation_loss/len(xvalid))
            if epoch%self.delta_epoch==0:
                print('Epoch [{}/{}], Train Loss: {:.5f}, Val Loss: {:.5f}'.format(epoch,self.epochs,epoch_loss/len(xtrain),validation_loss/len(xvalid)))
        traintime = (time() - start)/60                                          #end training timer
        n_params  = sum(p.numel() for p in self.model.parameters())              #count ROM # of parameters
        if self.verbose:
            print('# Parameters: {:,} | Training time: {:.3f} minutes\n'.format(n_params, traintime))
        if self.save_data:
            torch.save(self.model.state_dict(), 'h2_cushion_rom.pt')
        # predictions
        self.X_test_tensor  = torch.Tensor(self.X_test).to(self.device)        #tensorize X_test to gpu
        self.y_train_pred = np.array(self.model(self.X_train_tensor).tolist()) #predict and de-tensorize y_train_pred
        self.y_test_pred  = np.array(self.model(self.X_test_tensor).tolist())  #predict and de-tensorize y_test_pred
        if self.return_data:
            return self.y_train_pred, self.y_test_pred

    #################### PROCESSING ####################
    def read_data(self, n_subsample=None):
        '''
        Import data from the 4 main CSV files [ch4, co2, n2, nocg].
        '''
        data_ch4  = pd.read_csv('data/data_CH4.csv',  index_col=0)      #CH4  cushion gas dataset
        data_co2  = pd.read_csv('data/data_CO2.csv',  index_col=0)      #CO2  cushion gas dataset
        data_n2   = pd.read_csv('data/data_N2.csv',   index_col=0)      #N2   cushion gas dataset
        data_nocg = pd.read_csv('data/data_NOCG.csv', index_col=0)      #NONE cushion gas dataset 
        data_all  = pd.concat([data_ch4, data_co2, data_n2, data_nocg]) #collect into a single dataframe
        if n_subsample:
            idx = np.random.randint(0, data_all.shape[0], n_subsample)  #subsampling index
            self.all_data = data_all.iloc[idx,:]                        #subsample dataframe
        else:
            self.all_data = data_all
        if self.verbose:
            print('---------------- DATA INFORMATION ---------------')
            print('CH4: {} | CO2: {} | N2: {} | NOCG: {}'.format(data_ch4.shape, data_co2.shape, data_n2.shape, data_nocg.shape))
        if self.return_data:
            return self.all_data

    def process_data(self, restype='SA'):
        '''
        Process the data to: (1) shuffle, (2) truncate at large gwrt, (3) remove outliers, 
        (4) log-transform gwrt, (5) min-max scale, (6) train/test split.
        '''
        if self.noise_flag:
            data = self.all_data + np.random.normal(self.noise[0], self.noise[1], (self.all_data.shape)) #add random noise
        else:
            data = self.all_data
        data_shuffle = data.sample(frac=1)                                    #shuffle dataset
        data_trunc = data_shuffle[data_shuffle['gwrt']<self.gwrt_cutoff]      #truncate at gwrt threshold
        data_outl = data_trunc[(np.abs(zscore(data_trunc))<self.std_outlier)] #remove outliers
        if restype=='SA':
            data_outl.loc[:,'Sw'] = 1                                         #if SA, let all Sw=1
        self.data_clean = data_outl.dropna()                                  #drop nan's
        X_data, y_data = self.data_clean.iloc[:, self.xcols], self.data_clean.iloc[:, self.ycols] #split (X,y)
        y_data_log = pd.DataFrame(columns=y_data.columns)
        y_data_log['efft'], y_data_log['ymft'] = y_data['efft'], y_data['ymft']
        y_data_log['gwrt'] = np.log10(y_data['gwrt'])                         #log-transform gwrt
        self.x_scaler, self.y_scaler = MinMaxScaler(), MinMaxScaler()
        self.x_scaler.fit(X_data);   self.y_scaler.fit(y_data_log)            #Min-Max scaler [0,1]
        self.X_dataset = self.x_scaler.transform(X_data)
        self.y_dataset = self.y_scaler.transform(y_data_log)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_dataset, self.y_dataset, test_size=self.test_size)
        if self.verbose:
            print('Full dataset shape:', self.all_data.shape)
            print('Truncated dataset shape:', data_trunc.shape)
            print('Clean [no outliers] dataset shape:', self.data_clean.shape)
            print('Dataset: X={} | y={}'.format(self.X_dataset.shape, self.y_dataset.shape))
            print('X_train: {} | y_train: {}\nX_test:  {}  | y_test:  {}\n'.format(self.X_train.shape, self.X_test.shape, self.y_train.shape, self.y_test.shape))
        if self.save_data:
            np.save('data/X_train.npy', self.X_train); np.save('data/X_test.npy', self.X_test)
            np.save('data/y_train.npy', self.y_train); np.save('data/y_test.npy', self.y_test)
        if self.return_data:
            datasets = {'X_dataset':self.X_dataset, 'y_dataset':self.y_dataset}
            train_test_data = {'X_train':self.X_train, 'X_test':self.X_test, 'y_train':self.y_train, 'y_test':self.y_test}
            scalers = {'x_scaler':self.x_scaler, 'y_scaler':self.y_scaler}
            return datasets, train_test_data, scalers 

    def load_data(self):
        '''
        Directly load (X_train, X_test, y_train, y_test) from pre-saved npy files.
        '''
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
    
    #################### PLOTS & METRICS ####################
    def print_metrics(self):
        '''
        Compute and print performance metrics: R^2, MSE, MAE. This is done for the overall dataset, as well as per-target basis.
        '''
        self.tot_train_r2,  self.tot_test_r2  = r2_score(self.y_train, self.y_train_pred),  r2_score(self.y_test, self.y_test_pred)
        tot_train_mse, tot_test_mse = mean_squared_error(self.y_train, self.y_train_pred),  mean_squared_error(self.y_test, self.y_test_pred)
        tot_train_mae, tot_test_mae = mean_absolute_error(self.y_train, self.y_train_pred), mean_absolute_error(self.y_test, self.y_test_pred)
        print('-------------- PERFORMANCE METRICS --------------')
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
        print('-------------------------------------------------')
        print('---------------------- DONE ---------------------')
        print('-------------------------------------------------')
        if self.save_data:
            metrics = np.array([self.tot_train_r2, self.tot_test_r2, tot_train_mse, tot_test_mse, tot_train_mae, tot_test_mae])
            np.save('data/metrics.npy', metrics)

    def plot_loss(self, title='', figsize=(4,3)):
        '''
        Plot the training performance (loss per epoch).
        '''
        if figsize:
            plt.figure(figsize=figsize)
        loss, val = self.metrics['loss'], self.metrics['validation_loss']
        epochs = len(loss)
        iterations = np.arange(epochs)
        plt.plot(iterations, loss, '-', label='loss')
        plt.plot(iterations, val,  '-', label='validation loss')
        plt.title(title+' Training Loss vs. Epochs'); plt.legend()
        plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.xticks(iterations[::epochs//10])
        plt.savefig('figures/training_performance.png')
        if self.return_plot:
            plt.show()

    def plot_results(self, figsize=(15,4), figname='results'):
        '''
        Cross-plot of true-vs-predicted values for each of the targets [efft, ymft, gwrt]
        '''
        plt.figure(figsize=figsize)
        plt.suptitle(figname + ' -- Train $R^2$={:.3f} | Test $R^2$={:.3f}'.format(self.tot_train_r2, self.tot_test_r2))
        for i in range(3):
            name = self.y_labels[i]
            r2train = r2_score(self.y_train[:,i], self.y_train_pred[:,i])
            r2test  = r2_score(self.y_test[:,i],  self.y_test_pred[:,i])
            plt.subplot(1,3,i+1)
            plt.scatter(self.y_train[:,i], self.y_train_pred[:,i], alpha=0.25, label='train', edgecolor='tab:blue')
            plt.scatter(self.y_test[:,i],  self.y_test_pred[:,i],  alpha=0.25, label='test', edgecolor='tab:orange')
            plt.axline([0,0],[1,1], c='r', linewidth=3); plt.legend()
            plt.xlabel('True'); plt.ylabel('Predicted'); plt.xlim([-0.1,1.1]); plt.ylim([-0.1,1.1])
            plt.title('{} - $R^2_{}$={:.3f} ; $R^2_{}$={:.3f}'.format(name,'{train}',r2train,'{test}',r2test))
        plt.savefig('figures/' + figname + '.png')
        if self.return_plot:
            plt.show()

########################################################################################################################
############################################################ END #######################################################