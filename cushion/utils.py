########################################################################################################################
###################################################### IMPORT PACKAGES #################################################
########################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import keras
from keras.backend import clear_session
import tensorflow as tf
from keras import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.optimizers import Nadam
from keras.regularizers import L1, L2, L1L2

def check_tensorflow_gpu():
    sys_info = tf.sysconfig.get_build_info()
    print('Checking Tensorflow Version:')
    print('Tensorflow built with CUDA?',  tf.test.is_built_with_cuda())
    print('Tensorflow version:', tf.__version__)
    print('# GPU available:', len(tf.config.experimental.list_physical_devices('GPU')))
    print("CUDA: {} | cuDNN: {}".format(sys_info["cuda_version"], sys_info["cudnn_version"]))
    print(tf.config.list_physical_devices())
    return None

class h2proxy:
    def __init__(self):
        self.returns     = False
        self.verbose     = True
        self.NN_verbose  = False
        self.xcols       = range(12)
        self.ycols       = [12, 14, 15]
        self.noise_flag  = False
        self.noise       = [0, 0.05]
        self.gwrt_cutoff = 1e5
        self.std_outlier = 3
        self.test_size   = 0.3
        self.reg         = L1L2(1e-5)
        self.slope       = 0.33
        self.drop        = 0.1
        self.optim       = Nadam(1e-3)
        self.loss        = 'mae'
        self.metrics     = ['mae','mse']
        self.epochs      = 100
        self.batch_size  = 150
        self.valid_split = 0.2
        self.save_results   = True
        
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
            print('CH4: {} | CO2: {} | N2: {}'.format(data_ch4.shape, data_co2.shape, data_n2.shape))
            print('All: {}'.format(self.all_data.shape))
        if self.returns:
            return self.all_data

    def process_data(self):
        if self.noise_flag:
            data = self.all_data + np.random.normal(self.noise[0], self.noise[1], 
                                                    (self.all_data.shape[0], self.all_data.shape[1]))
        else:
            data = self.all_data
        data_trunc = data[data['gwrt']<self.gwrt_cutoff]
        data_clean = data_trunc[(np.abs(zscore(data_trunc))<self.std_outlier)]
        X_data, y_data = data_clean.iloc[:, self.xcols], data_clean.iloc[:, self.ycols]
        y_data_log = y_data.copy()
        y_data_log['gwrt'] = np.log10(y_data['gwrt'])
        self.x_scaler, self.y_scaler = MinMaxScaler(), MinMaxScaler()
        self.x_scaler.fit(X_data);   self.y_scaler.fit(y_data_log)
        X_norm = pd.DataFrame(self.x_scaler.transform(X_data), columns=X_data.columns)
        y_norm = pd.DataFrame(self.y_scaler.transform(y_data_log), columns=y_data.columns)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_norm, y_norm, test_size=self.test_size )
        if self.verbose:
            print('Full dataset shape:', self.all_data.shape)
            print('Truncated dataset shape:', data_trunc.shape)
            print('Clean (no outliers) dataset shape:', data_clean.shape)
            print('Train: X={} | y={}'.format(self.X_train.shape, self.y_train.shape))
            print('Test:  X={} | y={}'.format(self.X_test.shape,  self.y_test.shape)) 
        if self.returns:
            return self.X_train, self.X_test, self.y_train, self.y_test, self.x_scaler, self.y_scaler
    
    #################### ROM ####################   
    def make_model(self):
        inp = Input(shape=(self.X_train.shape[-1]))
        _ = Dense(48, activity_regularizer=self.reg)(inp)
        _ = LeakyReLU(self.slope)(_)
        _ = Dropout(self.drop)(_)
        _ = BatchNormalization()(_)
        
        _ = Dense(96, activity_regularizer=self.reg)(_)
        _ = LeakyReLU(self.slope)(_)
        _ = Dropout(self.drop)(_)
        _ = BatchNormalization()(_)
        
        _ = Dense(128, activity_regularizer=self.reg)(_)
        _ = LeakyReLU(self.slope)(_)
        _ = Dropout(self.drop)(_)
        _ = BatchNormalization()(_)
        
        _ = Dense(96, activity_regularizer=self.reg)(_)
        _ = LeakyReLU(self.slope)(_)
        _ = Dropout(self.drop)(_)
        _ = BatchNormalization()(_)
        
        _ = Dense(48, activity_regularizer=self.reg)(_)
        _ = LeakyReLU(self.slope)(_)
        _ = Dropout(self.drop)(_)
        _ = BatchNormalization()(_)
        
        _ = Dense(12, activity_regularizer=self.reg)(_)
        _ = LeakyReLU(self.slope)(_)
        _ = Dropout(self.drop)(_)
        _ = BatchNormalization()(_)
        
        out = Dense(self.y_train.shape[-1])(_)
        self.model = Model(inp, out, name='H2_ROM')   
        self.model.compile(optimizer=self.optim, loss=self.loss, metrics=self.metrics)
        self.fit = self.model.fit(self.X_train, self.y_train,
                                    epochs           = self.epochs,
                                    batch_size       = self.batch_size,
                                    validation_split = self.valid_split,
				                    shuffle          = True,
                                    verbose          = self.NN_verbose)
        if self.returns:
            return self.model, self.fit
    
    def make_predictions(self):
        self.y_train_pred = pd.DataFrame(self.model.predict(self.X_train), columns=self.y_train.columns)
        self.y_test_pred  = pd.DataFrame(self.model.predict(self.X_test),  columns=self.y_test.columns)
        if self.returns:
            return self.y_train_pred, self.y_test_pred
    
    #################### PLOTS & PRINTS ####################
    def rescale_results(self):
        def inv_scale(data):
            return pd.DataFrame(self.y_scaler.inverse_transform(data), columns=data.columns)
        self.y_train_r,      self.y_test_r      = inv_scale(self.y_train),      inv_scale(self.y_test)
        self.y_train_pred_r, self.y_test_pred_r = inv_scale(self.y_train_pred), inv_scale(self.y_test_pred)
        self.y_train_r['gwrt'], self.y_train_pred_r['gwrt'] = 10**self.y_train_r['gwrt'], 10**self.y_train_pred_r['gwrt']
        self.y_test_r['gwrt'],  self.y_test_pred_r['gwrt']  = 10**self.y_test_r['gwrt'],  10**self.y_test_pred_r['gwrt']
        if self.returns:
            return self.y_train_r, self.y_test_r, self.y_train_pred_r, self.y_test_pred_r
    
    def print_metrics(self):
        tot_train_r2,  tot_test_r2  = r2_score(self.y_train, self.y_train_pred),            r2_score(self.y_test, self.y_test_pred)
        tot_train_mse, tot_test_mse = mean_squared_error(self.y_train, self.y_train_pred),  mean_squared_error(self.y_test, self.y_test_pred)
        tot_train_mae, tot_test_mae = mean_absolute_error(self.y_train, self.y_train_pred), mean_absolute_error(self.y_test, self.y_test_pred)
        print('\n')
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
        if self.save_results:
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