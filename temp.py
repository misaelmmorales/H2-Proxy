#############################################################################################################################
###################################################### IMPORT PACKAGES ######################################################
#############################################################################################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
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

#############################################################################################################################
###################################################### PLOTS & PRINTS #######################################################
#############################################################################################################################
def plot_loss(fit, title='', figsize=None):
    if figsize:
        plt.figure(figsize=figsize)
    loss, val = fit.history['loss'], fit.history['val_loss']
    epochs = len(loss)
    iterations = np.arange(epochs)
    plt.plot(iterations, loss, '-', label='loss')
    plt.plot(iterations, val, '-', label='validation loss')
    plt.title(title+' Training: Loss vs epochs'); plt.legend()
    plt.ylabel('Epochs'); plt.ylabel('Loss'); plt.xticks(iterations[::epochs//10])
    plt.show()
    
def print_metrics(train, trainpred, test, testpred):
    tot_train_r2,  tot_test_r2  = r2_score(train, trainpred),            r2_score(test, testpred)
    tot_train_mse, tot_test_mse = mean_squared_error(train, trainpred),  mean_squared_error(test, testpred)
    tot_train_mae, tot_test_mae = mean_absolute_error(train, trainpred), mean_absolute_error(test, testpred)
    print('\n')
    print('TRAIN: R2={:.3f} | MSE={:.5f} | MAE={:.5f}'.format(tot_train_r2, tot_train_mse, tot_train_mae))
    print('TEST:  R2={:.3f} | MSE={:.5f} | MAE={:.5f}'.format(tot_test_r2, tot_test_mse, tot_test_mae))
    for i in range(3):
        name = train.columns[i]
        trainr2,  testr2  = r2_score(train[name], trainpred[name]),            r2_score(test[name], testpred[name])
        trainmse, testmse = mean_squared_error(train[name], trainpred[name]),  mean_squared_error(test[name], testpred[name])
        trainmae, testmae = mean_absolute_error(train[name], trainpred[name]), mean_absolute_error(test[name], testpred[name])
        print('\n')
        print('{} TRAIN: R2={:.3f}    | TEST: R2={:.3f}'.format(name, trainr2, testr2))
        print('{} TRAIN: MSE={:.5f} | TEST: MSE={:.5f}'.format(name, trainmse, testmse))   
        print('{} TRAIN: MAE={:.5f} | TEST: MAE={:.5f}'.format(name, trainmae, testmae))
        
def plot_results(train, trainpred, test, testpred, figsize=(15,4), figname='figure'):
    plt.figure(figsize=figsize)
    plt.suptitle(figname)
    for i in range(3):
        name = train.columns[i]
        plt.subplot(1,3,i+1)
        plt.axline([0,0],[1,1], c='r', linewidth=3)
        plt.scatter(train.iloc[:,i], trainpred.iloc[:,i], alpha=0.5, label='train')
        plt.scatter(test.iloc[:,i],  testpred.iloc[:,i],  alpha=0.3, label='test')
        plt.xlabel('True'); plt.ylabel('Predicted'); plt.legend()
        plt.title('{}'.format(name))
    plt.savefig(figname + '.png')
    plt.show()
    
#############################################################################################################################
######################################################## PROCESSING #########################################################
#############################################################################################################################

def process_data(data):
    print('Full dataset shape:', data.shape)
    data_trunc = data[data['gwrt']<1e5]
    print('Truncated dataset shape:', data_trunc.shape)
    data_clean = data_trunc[(np.abs(stats.zscore(data_trunc))<3).all(axis=1)]
    print('Clean (no outliers) dataset shape:', data_clean.shape)
    X_data = data_clean.iloc[:, 1:11]
    y_data = data_clean.iloc[:, [11,13,14]]
    y_data_log = y_data.copy()
    y_data_log['gwrt'] = np.log10(y_data['gwrt'])
    x_scaler, y_scaler = MinMaxScaler(), MinMaxScaler()
    x_scaler.fit(X_data)
    y_scaler.fit(y_data_log)
    X_norm = pd.DataFrame(x_scaler.transform(X_data), columns=X_data.columns)
    y_norm = pd.DataFrame(y_scaler.transform(y_data_log), columns=y_data.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y_norm, test_size=0.3)
    print('\n')
    print('Train: X={} | y={}'.format(X_train.shape, y_train.shape))
    print('Test:  X={} | y={}'.format(X_test.shape,  y_test.shape))
    
    return X_train, X_test, y_train, y_test, x_scaler, y_scaler

def rescale_results(train, trainpred, test, testpred, yscaler):
    train_r = pd.DataFrame(yscaler.inverse_transform(train), columns=train.columns)
    test_r  = pd.DataFrame(yscaler.inverse_transform(test),  columns=test.columns)
    train_pred_r = pd.DataFrame(yscaler.inverse_transform(trainpred), columns=trainpred.columns)
    test_pred_r  = pd.DataFrame(yscaler.inverse_transform(testpred),  columns=testpred.columns)
    train_r['gwrt'] = 10**train_r['gwrt']
    train_pred_r['gwrt'] = 10**train_pred_r['gwrt']
    test_r['gwrt'] = 10**test_r['gwrt']
    test_pred_r['gwrt'] = 10**test_pred_r['gwrt']
    return train_r, train_pred_r, test_r, test_pred_r

#############################################################################################################################
############################################################ ROM ############################################################
#############################################################################################################################
def make_model():
    clear_session()
    def dense_block(inp, neurons, slope=0.33, drop=0.10):
        _ = Dense(neurons, activity_regularizer=L1L2(1e-5))(inp)
        _ = LeakyReLU(slope)(_)
        _ = Dropout(drop)(_)
        _ = BatchNormalization()(_)
        return _
    inp = Input(shape=(10))
    _ = dense_block(inp, 50)
    _ = dense_block(_, 100)
    _ = dense_block(_, 150)
    _ = dense_block(_, 50)
    _ = dense_block(_, 10)
    out = Dense(3, activation='linear')(_)
    return Model(inp, out)

def train_model(model, xtrain, ytrain, 
                optimizer=Nadam(1e-3), loss='mae', metrics=['mse','mae'], 
                epochs=100, batch_size=100, validation_split=0.2, verbose=0):
    model.compile(optimizer = optimizer,
                  loss      = loss,
                  metrics   =  metrics)
    fit = model.fit(xtrain, ytrain,
                    epochs           = epochs,
                    batch_size       = batch_size,
                    validation_split = validation_split,
                    verbose          = verbose)
    return model, fit

def make_predictions(model, xtrain, xtest, ytrain, ytest):
    ytrain_pred = pd.DataFrame(model.predict(xtrain), columns=ytrain.columns)
    ytest_pred  = pd.DataFrame(model.predict(xtest),  columns=ytest.columns)
    return ytrain_pred, ytest_pred

#############################################################################################################################
############################################################ RUN ############################################################
#############################################################################################################################
check_tensorflow_gpu()
data = pd.read_csv('data.csv')

X_train, X_test, y_train, y_test, x_scaler, y_scaler = process_data(data)
model = make_model()
model, fit = train_model(model, X_train, y_train)
plot_loss(fit, figsize=(5,3))
y_train_pred, y_test_pred = make_predictions(model, X_train, X_test, y_train, y_test)
print_metrics(y_train, y_train_pred, y_test, y_test_pred)
y_train_r, y_train_pred_r, y_test_r, y_test_pred_r = rescale_results(y_train, y_train_pred, y_test, y_test_pred, y_scaler)
plot_results(y_train, y_train_pred, y_test, y_test_pred, figname='MinMax-Scaled')
plot_results(y_train_r, y_train_pred_r, y_test_r, y_test_pred_r, figname='Original Scale')

############################################################ END ############################################################
