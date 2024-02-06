import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

n_realizations = 1000
n_workers      = 20

def process_realization_F(i):
    data = np.load('train_dataF/realization_{}.npz'.format(i))
    x, y = data['X_data'], data['y_data']
    poro, perm, tstep = x[:,0], x[:,1], x[:,2]
    pres, satu = y[:,0], y[:,1]
    poro_n = np.expand_dims(cv2.resize(poro.T, (64,64)).T, 1)
    perm_n = np.expand_dims(cv2.resize(perm.T, (64,64)).T, 1)
    time_n = np.expand_dims(tstep[:,:64,:64], 1)
    pres_n = np.expand_dims(cv2.resize(pres.T, (64,64)).T, 1)
    satu_n = np.expand_dims(cv2.resize(satu.T, (64,64)).T, 1)
    X_data = np.concatenate([poro_n, perm_n, time_n], axis=1)
    y_data = np.concatenate([pres_n, satu_n], axis=1)
    np.savez('train_dataF_64x64/realization_{}'.format(i), X_data=X_data, y_data=y_data)

with ThreadPoolExecutor(max_workers=n_workers) as executor:
    executor.map(process_realization_F, range(1,n_realizations+1))