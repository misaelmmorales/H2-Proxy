################################################################
####################### H2 Heterogeneity #######################
################# Misael Morales & Shaowen Mao #################
################# Los Alamos National Laboratoy ################
########################## Summer 2023 #########################
################################################################
from utils import *

hete = Heterogeneity()
hete.check_torch_gpu()

facies = hete.load_facies()
#hete_fluv, hete_gaus = hete.load_perm_poro()

################################################################

fdir, gdir = os.path.join(hete.data_dir, 'h2dataf'), os.path.join(hete.data_dir, 'h2datag')

n_realizations = 1000

X_data = np.zeros((2*n_realizations,61,256,256,4))
for i in range(n_realizations):
    t = np.load('data2D/times/time_{}.npy'.format(i))
    # Fluvial
    dataf = loadmat(fdir + '/{}UHSS_0'.format(i+1))
    porof = dataf['PORO']
    permf = np.log10(dataf['PERMX'])
    chanf = np.moveaxis(facies[i].T, 0, -1)
    Xf = np.repeat(np.concatenate([porof, permf, chanf], -1)[np.newaxis,...], 61, 0)
    Xf = np.concatenate([Xf, t], -1)
    X_data[i] = Xf
    # Gaussian
    datag = loadmat(gdir + '/{}UHSS_0'.format(i+1))
    porog = datag['PORO']
    permg = np.log10(datag['PERMX'])
    angle = np.load('data2D/angles/angle_{}.npy'.format(i))
    Xg = np.repeat(np.concatenate([porog, permg, angle], -1)[np.newaxis,...], 61, 0)
    Xg = np.concatenate([Xg, t], -1)
    X_data[i+n_realizations] = Xg
print(X_data.shape)
X_data = np.moveaxis(X_data, -1, 2).reshape(2*n_realizations*61,4,256,256)
print(X_data.shape)

y_data = np.zeros((2*n_realizations,61,256,256,2))
for i in range(n_realizations):
    for j in range(61):
        dataf = loadmat(fdir + '/{}UHSS_{}'.format(i+1,j))
        datag = loadmat(gdir + '/{}UHSS_{}'.format(i+1,j))
        y_data[i,j,:,:,0] = dataf['PRESSURE']
        y_data[i,j,:,:,1] = dataf['SGAS'] * dataf['YMF_3']  
        y_data[i+n_realizations,j,:,:,0] = datag['PRESSURE']
        y_data[i+n_realizations,j,:,:,1] = datag['SGAS'] * datag['YMF_3']
print(y_data.shape)
y_data = np.moveaxis(y_data.reshape(2*n_realizations*61,256,256,2),-1,1)
print(y_data.shape)

np.save('X_data.npy', X_data)
np.save('y_data.npy', y_data)