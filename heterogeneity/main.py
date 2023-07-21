################################################################
####################### H2 Heterogeneity #######################
################# Misael Morales & Shaowen Mao #################
################# Los Alamos National Laboratoy ################
########################## Summer 2023 #########################
################################################################
from utils import *
from HeteTrans import *

hete = Heterogeneity()
hete.check_torch_gpu()

################################################################
nr = hete.n_realizations
nt = 61
ni = 64
nx = 4
ny = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 102
lr         = 2e-4
num_epochs = 100

################################################################
X_data, y_data = hete.load_xy(200)
print('X_data: {} | y_data: {}'.format(X_data.shape, y_data.shape))

X_tr,    X_test,  y_tr,    y_test  = train_test_split(X_data, y_data, test_size=0.20)
X_train, X_valid, y_train, y_valid = train_test_split(X_tr, y_tr, test_size=0.25)
print('X_train: {} | X_valid: {} | X_test: {}'.format(X_train.shape, X_valid.shape, X_test.shape))
print('y_train: {} | y_valid: {} | y_test: {}'.format(y_train.shape, y_valid.shape, y_test.shape))
n_train, n_valid, n_test = X_train.shape[0], X_valid.shape[0], X_test.shape[0]
print('Number of: train={}, valid={}, test={}'.format(n_train, n_valid, n_test))

xtr = np.moveaxis(np.moveaxis(X_train, -2, 1).reshape(n_train*nt,ni,ni,nx), -1, 1)
ytr = np.moveaxis(np.moveaxis(y_train, -2, 1).reshape(n_train*nt,ni,ni,ny), -1, 1)
print('X_train_reshape: {} | y_train_reshape: {}'.format(xtr.shape, ytr.shape))
train_dataset    = NumpyDataset_from_array(xtr, ytr)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

xvl = np.moveaxis(np.moveaxis(X_valid, -2, 1).reshape(n_valid*nt,ni,ni,nx), -1, 1)
yvl = np.moveaxis(np.moveaxis(y_valid, -2, 1).reshape(n_valid*nt,ni,ni,ny), -1, 1)
print('X_valid_reshape: {} | y_valid_reshape: {}'.format(xvl.shape, yvl.shape))
valid_dataset    = NumpyDataset_from_array(xvl, yvl)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

xte = np.moveaxis(np.moveaxis(X_test, -2, 1).reshape(n_test*nt,ni,ni,nx), -1, 1)
yte = np.moveaxis(np.moveaxis(y_test, -2, 1).reshape(n_test*nt,ni,ni,ny), -1, 1)
print('X_test_reshape: {} | y_test_reshape: {}'.format(xte.shape, yte.shape))
test_dataset    = NumpyDataset_from_array(xte, xte)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

################################################################
model     = VisionTransformer().to(device)
criterion = CustomLoss(0.5, 0.5)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (X, Y) in enumerate(train_dataloader):
        inp, out = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(inp)
        loss = criterion(pred, out)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_dataloader)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for j, (Xv, Yv) in enumerate(valid_dataloader):
            inpv, outv = Xv.to(device), Yv.to(device)
            predv = model(inpv)
            val_loss += criterion(predv, outv).item()
    avg_val_loss = val_loss / len(valid_dataloader)
    print('Epoch: [{}/{}]: Loss={:.3f}, Val Loss={:.3f}'.format(epoch+1, num_epochs, avg_loss, avg_val_loss))
    
pd.DataFrame({'loss':avg_loss, 'val_loss':avg_val_loss}).to_csv('metrics.csv')
torch.save(model.state_dict(), 'HeteTrans.pt')