################################################################
####################### H2 Heterogeneity #######################
################# Misael Morales & Shaowen Mao #################
################# Los Alamos National Laboratoy ################
########################## Summer 2023 #########################
################################################################
from utils import *

hete = Heterogeneity()
hete.check_torch_gpu()

################################################################
facies = hete.load_facies()

############################# END ##############################

X_data, y_data = hete.load_xy()
x = np.moveaxis(np.moveaxis(X_data, -2, 1).reshape(X_data.shape[0]*61,64,64,4), -1, 1)
y = np.moveaxis(np.moveaxis(y_data, -2, 1).reshape(X_data.shape[0]*61,64,64,2), -1, 1)
print(x.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)
print('X_train: {} | y_train: {}\nX_test:  {} | y_test:  {}'.format(X_train.shape, y_train.shape, X_test.shape,  y_test.shape))

device = 'cuda'

rom = h2_hete_rom().to(device)
optimizer = NAdam(rom.parameters(), lr=1e-3, weight_decay=1e-6)
loss_fn = nn.MSELoss()
loss, val_loss = [], []
metrics = {'loss':[], 'val_loss':[]}
epochs = 50
batch_size = 100
for epoch in range(epochs):
    rom.train()
    epoch_loss = 0.0
    for i in range(0, len(X_train), batch_size):
        inp  = torch.Tensor(X_train[i:i+batch_size]).to(device)
        true = torch.Tensor(y_train[i:i+batch_size]).to(device)
        optimizer.zero_grad()
        pred = rom(inp)
        loss = loss_fn(pred,true)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*inp.size(0)
    metrics['loss'].append(epoch_loss/len(X_train)) 
plt.figure(figsize=(4,3))
plt.plot(metrics['loss'], label='loss')
plt.legend(); plt.grid('on')
plt.title('Training Loss'); plt.xlabel('epoch'); plt.ylabel('loss')
plt.savefig('training_performance.png')
plt.show()

vis_realization = 411
x_true = np.moveaxis(np.moveaxis(X_data[vis_realization], -2, 0), -1, 1)
y_true = np.moveaxis(np.moveaxis(y_data[vis_realization], -2, 0), -1, 1)
print(x_true.shape, y_true.shape)
y_pred = rom(torch.Tensor(x_true).to(device)).cpu().detach().numpy()
print(y_pred.shape)
labels = ['$\phi$', '$P$', '$\hat{P}$', '$|P-\hat{P}|$', '$S$', '$\hat{S}$', '$|S-\hat{S}|$']
fig, axs = plt.subplots(7, 12, figsize=(20,8))
for j in range(12):   
    axs[0,j].imshow(x_true[-1,0], 'jet'); axs[0,j].set(title='R{} t{}'.format(vis_realization, j*5))
    p,  s  = y_true[j*5,0], y_true[j*5,1]
    p_, s_ = y_pred[j*5,0], y_pred[j*5,1]
    axs[1,j].imshow(p, 'jet')               ; axs[4,j].imshow(s, 'jet')
    axs[2,j].imshow(p_, 'jet')              ; axs[5,j].imshow(s_, 'jet')
    axs[3,j].imshow(np.abs(p-p_), 'binary') ; axs[6,j].imshow(np.abs(s-s_), 'binary')
    for i in range(7):
        axs[i,j].set(xticks=[], yticks=[])
for i in range(7):
    axs[i,0].set(ylabel=labels[i])
plt.savefig('results_{}.png'.format(vis_realization))
plt.show() 