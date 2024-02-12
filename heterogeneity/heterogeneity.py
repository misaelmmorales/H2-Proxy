import os, time, math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2 as transforms
from torchvision.ops import SqueezeExcitation
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class H2ViT:
    def __init__(self):
        self.verbose        = True                # print progress
        self.return_data    = False               # return data
        self.folder         = 'train_dataF_64x64' # dataset directory (F=fluvial, G=gaussian)
        self.check_torch_gpu()                    # check if torch is built with GPU support
        
        self.image_size     = 64                  # image size      (DO NOT CHANGE)
        self.in_channels    = 3                   # input channels  (DO NOT CHANGE)
        self.out_channels   = 2                   # output channels (DO NOT CHANGE)

        self.train_perc     = 0.70                # training set split percentage (of total)
        self.valid_perc     = 0.15                # validation set split percentage (of total)
        self.batch_size     = 25                  # batch size

        self.num_epochs     = 120                 # number of epochs
        self.monitor_step   = 10                  # monitoring training performance
        self.lr             = 1e-3                # learning rate
        self.weight_decay   = 1e-5                # weight decay for learning rate
        self.mse_weight     = 0.80                # Combined loss MSE weight
        self.ssim_weight    = 0.20                # Combined loss SSIM weight

        self.patch_size     = 16                  # patch size
        self.projection_dim = 128                 # projection dimension
        self.num_layers     = 3                   # number of layers
        self.num_heads      = 8                   # number of heads
        self.embed_dim      = 512                 # embedding dimension
        self.max_seq_len    = 1024                # maximum sequence length
        self.mlp_hidden_dim = 128                 # MLP hidden dimension

    def check_torch_gpu(self):
        '''
        Check if Torch is successfully built with GPU support
        '''
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        if self.verbose:
            print('\n'+'-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch Version {} | Torch Build with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('-'*60+'\n')
        self.device = torch.device('cuda' if cuda_avail else 'cpu')
        return None
    
    def count_params(self, model):
        ### Count the total number of trainable parameters in the neural network
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def load_data(self):
        '''
        Make the Train/Valid/Test dataloaders from custom Dataset and DataLoader classes
        '''
        print('-'*60+'\n'+'-------------- DATA LOADING AND PREPROCESSING --------------') if self.verbose else None
        x_transform = transforms.Compose([transforms.Normalize(mean=[0, 0, 0], std=[1,1,1])])
        y_transform = transforms.Compose([transforms.Normalize(mean=[0, 0], std=[1,1])])
        self.transforms = [x_transform, y_transform]
        dataset = CustomDataset(self.folder, x_transform, y_transform)
        train_size = int(self.train_perc * len(dataset))
        valid_size = int(self.valid_perc * len(dataset))
        test_size  = len(dataset) - train_size - valid_size
        train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size])
        self.train_loader = CustomDataloader(train_data, mode='train', batch_size=self.batch_size, shuffle=True, x_transform=x_transform, y_transform=y_transform)
        self.valid_loader = CustomDataloader(valid_data, mode='valid', batch_size=self.batch_size, shuffle=True, x_transform=x_transform, y_transform=y_transform)
        self.test_loader  = CustomDataloader(test_data,  mode='test',  batch_size=self.batch_size, shuffle=True, x_transform=x_transform, y_transform=y_transform)
        if self.verbose:
            print('Train size:   {} | Valid size:  {} | Test size:  {}'.format(train_size, valid_size, test_size))
            print('Train batches: {} | Valid batches: {} | Test batches: {}'.format(len(self.train_loader), len(self.valid_loader), len(self.test_loader)))
            print('-'*60+'\n')
        if self.return_data:
            return self.train_loader, self.valid_loader, self.test_loader
        
    def train_model(self, optimizer=None, criterion=None):
        '''
        Subroutine for training the model
        '''
        self.model = H2ViTnet(image_size=self.image_size, in_channels=self.in_channels, 
                              patch_size=self.patch_size, projection_dim=self.projection_dim, 
                              num_layers=self.num_layers, num_heads=self.num_heads, 
                              embed_dim=self.embed_dim,   max_seq_len=self.max_seq_len, 
                              mlp_hidden_dim=self.mlp_hidden_dim).to(self.device)
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if criterion is None:
            criterion = CustomLoss(mse_weight=self.mse_weight, ssim_weight=self.ssim_weight).to(self.device)
        if self.verbose:
            print('Total number of trainable parameters: {:,}'.format(self.count_params(self.model)))
        train_loss, valid_loss, time0 = [], [], time.time()
        for epoch in range(self.num_epochs):
            start_time = time.time()
            # Training
            self.model.train()
            epoch_loss = 0.0
            for i, (x,y) in enumerate(self.train_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            train_loss.append(epoch_loss/(i+1))
            # Validation
            self.model.eval()
            epoch_loss = 0.0
            with torch.no_grad():
                for i, (x,y) in enumerate(self.valid_loader):
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = self.model(x)
                    loss = criterion(y_pred, y)
                    epoch_loss += loss.item()
                valid_loss.append(epoch_loss/(i+1))
            # Save model and losses
            end_time = time.time() - start_time
            if self.verbose and (epoch+1) % self.monitor_step == 0:
                print('Epoch: {}/{} | Train loss: {:.4f} | Val loss: {:.4f} | Time elapsed: {:.2f} sec'.format(
                    epoch+1, self.num_epochs, train_loss[-1], valid_loss[-1], end_time))
        print('-'*60+'\n','Total training time: {:.2f} min'.format((time.time()-time0)/60), '\n'+'-'*60+'\n') if self.verbose else None
        print(' '*24+'... done ...'+' '*24+'\n'+'-'*60+'\n') if self.verbose else None
        self.losses = [train_loss, valid_loss]
        #torch.save(self.model.state_dict(), 'h2vit_model.pth')
        return self.model, self.losses if self.return_data else None
    
    def plot_losses(self, figsize=(5,4), showfig:bool=True):
        '''
        Plot the training and validation losses
        '''
        train_loss, valid_loss = self.losses
        plt.figure(figsize=figsize)
        plt.plot(train_loss, label='Train loss')
        plt.plot(valid_loss, label='Valid loss')
        plt.xlabel('Epochs', weight='bold'); plt.ylabel('Loss', weight='bold')
        plt.title('Training and Validation Losses', weight='bold')
        plt.legend(facecolor='lightgrey', edgecolor='k', fancybox=False)
        plt.tight_layout()
        plt.savefig('losses.png', dpi=600, bbox_inches='tight')
        plt.show() if showfig and self.verbose else None
        return None
    
    def plot_samples(self, figsize=(15,4)):
        for i, (x0, y0) in enumerate(self.train_loader):
            print(x0.shape, y0.shape)
            break
        def process(x, n_channels, to_cpu=False):
            if to_cpu:
                x1 = x.detach().cpu().numpy().reshape(self.batch_size, 10, n_channels, self.image_size, self.image_size)
            else:
                x1 = x.detach().numpy().reshape(self.batch_size, 10, n_channels, self.image_size, self.image_size)
            return x1
        yh = self.model(x0.to(self.device))
        x0 = process(x0, self.in_channels)
        y0 = process(y0, self.out_channels)
        yh = process(yh, self.out_channels, to_cpu=True)
        print('X0: {} | Y0: {} | Yh: {}'.format(x0.shape, y0.shape, yh.shape))
        fig, axs = plt.subplots(3, 10, figsize=figsize)
        for i in range(3):
            for j in range(10):
                axs[i,j].imshow(x0[i, j, 0], 'jet')
                axs[i,j].set(xticks=[], yticks=[])
        plt.tight_layout(); plt.savefig('x0.png')
        fig, axs = plt.subplots(3, 10, figsize=figsize)
        for i in range(3):
            for j in range(10):
                axs[i,j].imshow(y0[i, j, 1], 'jet')
                axs[i,j].set(xticks=[], yticks=[])
        plt.tight_layout(); plt.savefig('y0.png')
        fig, axs = plt.subplots(3, 10, figsize=figsize)
        for i in range(3):
            for j in range(10):
                axs[i,j].imshow(yh[i, j, 1], 'jet')
                axs[i,j].set(xticks=[], yticks=[])
        plt.tight_layout(); plt.savefig('yh.png')
        return None

#####################################################################################
########################### DATA LOADING AND PREPROCESSING ##########################
#####################################################################################
class CustomDataset(Dataset):
    def __init__(self, data_folder, x_transform=None, y_transform=None):
        self.data_folder = data_folder
        self.file_list   = os.listdir(data_folder)
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.transform   = True if x_transform and y_transform else False
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_folder, self.file_list[idx])
        data = np.load(file_path)
        X_data, y_data = torch.Tensor(data['X_data']), torch.Tensor(data['y_data'] )
        if self.transform:
            X_data = self.x_transform(X_data)
            y_data = self.y_transform(y_data)
        return X_data, y_data
    
class CustomDataloader(DataLoader):
    def __init__(self, *args, mode:str=None, x_transform=None, y_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode        = mode
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.transform   = True if x_transform and y_transform else False
    def __iter__(self):
        for batch in super().__iter__():
            X_data, y_data = batch
            if self.mode == 'train':
                X_data = X_data[:, :40]
                y_data = y_data[:, :40]
            elif self.mode == 'valid':
                X_data = X_data[:, 40:50]
                y_data = y_data[:, 40:50]
            elif self.mode == 'test':
                X_data = X_data[:, 50:]
                y_data = y_data[:, 50:]
            else:
                raise ValueError("mode must be either 'train, 'valid', or 'test'")
            X_data = X_data[:, ::X_data.shape[1]//10]
            y_data = y_data[:, ::y_data.shape[1]//10]
            X_data = X_data.reshape(-1, X_data.size(2), X_data.size(3), X_data.size(4)) # reshape to (b*t, c, h, w)
            y_data = y_data.reshape(-1, y_data.size(2), y_data.size(3), y_data.size(4)) # reshape to (b*t, c, h, w)
            if self.transform:
                X_data = self.x_transform(X_data)
                y_data = self.y_transform(y_data)
            yield X_data, y_data

#####################################################################################
################################### MODEL CLASSES ###################################
#####################################################################################
class CustomLoss(nn.Module):
    def __init__(self, mse_weight=1.0, ssim_weight=1.0):
        super(CustomLoss, self).__init__()
        self.mse_weight  = mse_weight
        self.ssim_weight = ssim_weight
        self.mse         = nn.MSELoss()
        self.ssim        = SSIM()
    def forward(self, y_pred, y_true):
        mse_loss  = self.mse(y_pred, y_true)
        ssim_loss = 1 - self.ssim(y_pred, y_true)
        return self.mse_weight*mse_loss + self.ssim_weight*ssim_loss
    
class PatchEmbedding(nn.Module):
    '''
    Patchify the input image into patches for vision transformer
    '''
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size                           # get original image size
        self.patch_size = patch_size                           # get user-defined patch size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        patches = self.projection(x)                           # convolve image to patch
        patches = rearrange(patches, 'b c h w -> b (h w) c')   # rearrange patches
        return patches
    
class PositionalEncoding(nn.Module):
    '''
    Get the positional codes for each patch of the input image
    '''
    def __init__(self, embed_dim, max_seq_len):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()   # get position indices
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pos_enc  = torch.zeros((1, max_seq_len, embed_dim))            # instantiate empty tensor
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)           # compute positional encoding sine
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)           # compute positional encoding cosine
        self.register_buffer('pos_enc', pos_enc)                       # register buffer for positional encoding
    def forward(self, x):
        x = x + self.pos_enc[:, :x.size(1)].detach()                   # add positional encoding to patches
        return x
    
class MultiHeadAttention(nn.Module):
    '''
    QKV MultiHead Attention mechanism
    '''
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads                                                         # required: embed_dim/num_heads
        self.query  = nn.Linear(embed_dim, embed_dim)                                                   # querys
        self.key    = nn.Linear(embed_dim, embed_dim)                                                   # keys
        self.value  = nn.Linear(embed_dim, embed_dim)                                                   # values
        self.fc_out = nn.Linear(embed_dim, embed_dim)                                                   # outputs
    def forward(self, query, key, value):
        batch_size = query.shape[0]
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # calculate query
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)       # calculate keys
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)   # calculate values
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)) # get scores
        attention_weights = F.softmax(scores, dim=-1)                                                   # get attention weights
        out = torch.matmul(attention_weights, V)                                                        # calculate output 
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)                 # rearrange output
        out = self.fc_out(out)                                                                          # compute output
        return out
    
class MLPBlock(nn.Module):
    '''
    Multi-Layer Perceptron block for vision transformer
    '''
    def __init__(self, embed_dim, mlp_hidden_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, embed_dim)
    def forward(self, x):
        x = F.gelu(self.fc1(x))   # activate outputs
        x = self.fc2(x)           # compute outputs
        return x
    
class TransformerEncoderBlock(nn.Module):
    '''
    Single ViT block with attention
    '''
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)   # Attention mechanism
        self.mlp_block = MLPBlock(embed_dim, mlp_hidden_dim)             # MLP block
        self.norm1 = nn.LayerNorm(embed_dim)                             # normalization 1
        self.norm2 = nn.LayerNorm(embed_dim)                             # normalization 2
    def forward(self, x):
        attention_output = self.self_attention(x, x, x)                  # attention values
        x = x + attention_output                                         # update attention
        x = self.norm1(x)                                                # normalize attention
        mlp_output = self.mlp_block(x)                                   # apply MLP block
        x = x + mlp_output                                               # update MLP outputs
        x = self.norm2(x)                                                # normalize outputs
        return x
    
class ViTencoder(nn.Module):
    '''
    Single ViT block with patch embedding and positional encoding
    '''
    def __init__(self, image_size, in_channels, latent_size, patch_size, projection_dim, 
                 num_layers, num_heads, embed_dim, max_seq_len, mlp_hidden_dim):
        super(ViTencoder, self).__init__()
        self.patch_embedding     = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)      # patch embedding
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len)                          # positional encoding
        self.transformer_blocks  = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_hidden_dim) for _ in range(num_layers)]) # N Transfromer blocks
        self.global_avg_pooling  = nn.AdaptiveAvgPool1d(1)                                             # global average pooling
        self.fc = nn.Linear(embed_dim, projection_dim*latent_size*latent_size)                         # fully connected layer
    def forward(self, x):
        x = self.patch_embedding(x)                                                                    # patch embedding
        x = self.positional_encoding(x)                                                                # positional encoding
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)                                                                   # transformer block(s)
        x = self.global_avg_pooling(x.transpose(1, 2))                                                 # global average pooling
        x = x.squeeze(2)                                                                               # squeeze output
        x = self.fc(x)                                                                                 # activate outputs
        return x
    
class H2ViTnet(nn.Module):
    def __init__(self, image_size=64, in_channels=3, patch_size=8, projection_dim=32, 
                 num_layers=2, num_heads=8, embed_dim=16, max_seq_len=256, mlp_hidden_dim=64):
        super(H2ViTnet, self).__init__()
        self.image_size     = image_size
        self.in_channels    = in_channels
        self.projection_dim = projection_dim
        self.latent_size    = self.image_size//8
        self.patch_size     = patch_size
        self.num_layers     = num_layers
        self.num_heads      = num_heads
        self.embed_dim      = embed_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.max_seq_len    = max_seq_len
        self.encoder = ViTencoder(image_size     = self.image_size, 
                                  in_channels    = self.in_channels,
                                  projection_dim = projection_dim, 
                                  latent_size    = self.latent_size,
                                  patch_size     = self.patch_size,
                                  num_heads      = self.num_heads,
                                  num_layers     = self.num_layers,
                                  embed_dim      = self.embed_dim,
                                  max_seq_len    = self.max_seq_len,
                                  mlp_hidden_dim = self.mlp_hidden_dim)
        self.layers = nn.Sequential(
            self._conv_block(projection_dim, projection_dim//2),
            self._conv_block(projection_dim//2, projection_dim//4),
            self._conv_block(projection_dim//4, self.latent_size),
        )
        self.out = nn.Conv2d(self.latent_size, 2, kernel_size=3, padding=1)
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            SqueezeExcitation(out_channels, out_channels//4),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(),
            nn.Upsample(scale_factor=2))
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.projection_dim, self.latent_size, self.latent_size)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)
    
#####################################################################################
######################################## MAIN #######################################
#####################################################################################
if __name__ == '__main__':
    h = H2ViT()
    h.load_data()
    h.train_model()
    h.plot_losses(showfig=False)
    h.plot_samples()
    torch.save(h.model.state_dict(), 'h2vit_model.pth')
    print(' '*24+'... done ...'+' '*24+'\n'+'-'*60+'\n') if h.verbose else None

#####################################################################################
#################################### END OF FILE ####################################
#####################################################################################