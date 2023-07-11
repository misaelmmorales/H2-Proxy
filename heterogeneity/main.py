################################################################
####################### H2 Heterogeneity #######################
################# Misael Morales & Shaowen Mao #################
################# Los Alamos National Laboratoy ################
########################## Summer 2023 #########################
################################################################
from utils import *

hete = Heterogeneity()
hete.check_torch_gpu()

#facies = hete.load_facies()
#hete_fluv, hete_gaus = hete.load_perm_poro()

################################################################

n_realizations = 5
n_timesteps    = 61
dim            = 256

pressure, sat_h2 = np.zeros((n_realizations,n_timesteps,dim,dim,1)), np.zeros((n_realizations,n_timesteps,dim,dim,1))
poro,     perm   = np.zeros((n_realizations,dim,dim,1)),             np.zeros((n_realizations,dim,dim,1))

for i in range(n_realizations):
    data0 = loadmat('//dcstorage.lanl.gov/MFR2/misael/h2dataf/{}UHSS_0'.format(i+1))
    poro[i], perm[i] = data0['PORO'], data0['PERMX']
    for j in range(n_timesteps):
        data = loadmat('//dcstorage.lanl.gov/MFR2/misael/h2dataf/{}UHSS_{}'.format(i+1,j))
        pressure[i,j] = data['PRESSURE']
        sat_h2[i,j]   = data['SGAS'] * data['YMF_3']
        
facies = np.expand_dims(hete.load_facies()[:n_realizations],-1)
print(facies.shape)

t_steps = np.arange(61)
times = np.ones((n_realizations,n_timesteps,dim,dim,1))
for i in range(n_timesteps):
    times[:,i] = times[:,i]*t_steps[i]
print(t_steps.shape, times.shape)

X_train = np.concatenate([np.repeat(np.concatenate([poro,perm,facies], -1)[:,np.newaxis,...], n_timesteps, axis=1),times],-1).reshape(n_realizations*n_timesteps,dim,dim,4)
y_train = np.concatenate([sat_h2, pressure],-1).reshape(n_realizations*n_timesteps,dim,dim,2)
X_train = np.moveaxis(X_train, -1, 1)
y_train = np.moveaxis(y_train, -1, 1)
print('X: {} | y: {}'.format(X_train.shape, y_train.shape))

device = 'cuda'

rom = h2_hete_rom().to(device)
optimizer = NAdam(rom.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

X_train = torch.Tensor(X_train).to(device)
y_train = torch.Tensor(y_train).to(device)

loss, val_loss = [], []
metrics = {'loss':[], 'val_loss':[]}
epochs = 10
batch_size = 50

for epoch in range(epochs):
    rom.train()
    epoch_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        inp  = X_train[i:i+batch_size]
        true = y_train[i:i+batch_size]
        optimizer.zero_grad()
        pred = rom(inp)
        loss = loss_fn(pred,true)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*inp.size(0)
    metrics['loss'].append(epoch_loss/len(X_train))
    
np.save('metrics.npy', np.array(list(metrics.values())))
dat = {'X':X_train, 'y':y_train, 'yhat':rom(X_train)}
np.save('dat.npy', np.array(list(dat.values())))