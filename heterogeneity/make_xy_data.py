from utils import *

hete = Heterogeneity()
hete.check_torch_gpu()
facies = hete.load_facies()

hete.data_dir = '/project/MFR2/misael'
fdir, gdir = os.path.join(hete.data_dir, 'h2dataf'), os.path.join(hete.data_dir, 'h2datag')

X_data = np.zeros((2*hete.n_realizations, 64, 64, 61, 4))
y_data = np.zeros((2*hete.n_realizations, 64, 64, 61, 2))

time_masks = np.moveaxis(repmat(np.expand_dims(np.arange(0,61), -1), 1, 64*64).reshape(61,64,64), 0, -1)/60
angle_masks = np.tile(np.repeat(np.expand_dims(repmat(np.expand_dims(np.array(hete.theta), -1), 1, 64*64).reshape(8,64,64), -1), 125, 0), 61)

pressf, pressg = np.zeros((64,64,61)), np.zeros((64,64,61))
saturf, saturg = np.zeros((64,64,61)), np.zeros((64,64,61))

for i in range(hete.n_realizations):
    tstep = time_masks
    
    # FLUVIAL
    dataf = loadmat(fdir + '/{}UHSS_0'.format(i+1))
    porof, permf, chanf = resize(dataf['PORO'], (64,64)), np.log10(resize(dataf['PERMX'], (64,64))), resize(facies[i].T, (64,64))
    porofn = MinMaxScaler().fit_transform(porof.reshape(-1,1)).reshape(64,64)
    permfn = MinMaxScaler().fit_transform(permf.reshape(-1,1)).reshape(64,64)
    chanfn = MinMaxScaler().fit_transform(chanf.reshape(-1,1)).reshape(64,64)
    phif = np.tile(porofn[...,np.newaxis], 61)
    kxxf = np.tile(permfn[...,np.newaxis], 61)
    facf = np.tile(chanfn[...,np.newaxis], 61)
    X_data[i] = np.concatenate([phif[...,np.newaxis], kxxf[...,np.newaxis], facf[...,np.newaxis], tstep[...,np.newaxis]], -1)
    # GAUSSIAN
    datag = loadmat(gdir + '/{}UHSS_0'.format(i+1))
    porog, permg = resize(datag['PORO'], (64,64)), np.log10(resize(datag['PERMX'], (64,64)))
    porogn = MinMaxScaler().fit_transform(porog.reshape(-1,1)).reshape(64,64)
    permgn = MinMaxScaler().fit_transform(permg.reshape(-1,1)).reshape(64,64)
    phig = np.tile(porogn[...,np.newaxis], 61)
    kxxg = np.tile(permgn[...,np.newaxis], 61)
    angle = angle_masks[i]/150
    X_data[i+hete.n_realizations] = np.concatenate([phig[...,np.newaxis], kxxg[...,np.newaxis], angle[...,np.newaxis], tstep[...,np.newaxis]], -1)
    
    for j in range(61):
        # FLUVIAL
        dataf = loadmat(fdir + '/{}UHSS_{}'.format(i+1,j))
        pressf[...,j] = resize(dataf['PRESSURE'], (64,64))
        saturf[...,j] = resize(dataf['SGAS'], (64,64)) * resize(dataf['YMF_3'], (64,64))
        pf = MinMaxScaler().fit_transform(pressf.reshape(64*64,61).T).T.reshape(64,64,61)
        sf = MinMaxScaler().fit_transform(saturf.reshape(64*64,61).T).T.reshape(64,64,61)
        # GAUSSIAN
        datag = loadmat(gdir + '/{}UHSS_{}'.format(i+1,j))
        pressg[...,j] = resize(datag['PRESSURE'], (64,64))
        saturg[...,j] = resize(datag['SGAS'], (64,64)) * resize(datag['YMF_3'], (64,64))
        pg = MinMaxScaler().fit_transform(pressg.reshape(64*64,61).T).T.reshape(64,64,61)
        sg = MinMaxScaler().fit_transform(saturg.reshape(64*64,61).T).T.reshape(64,64,61)  
    
    y_data[i] = np.concatenate([pf[...,np.newaxis], sf[...,np.newaxis]], -1)
    y_data[i+hete.n_realizations] = np.concatenate([pg[...,np.newaxis], sg[...,np.newaxis]], -1)
    if i%10==0:
        print('Realization {}(F,G) DONE!'.format(i))
    
print(X_data.shape, y_data.shape)
print('DONE!')
np.save('X_data.npy', X_data); np.save('y_data.npy', y_data)