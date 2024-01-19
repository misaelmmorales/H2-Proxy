import os, time, math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import v2 as TVTransform
from torchvision.ops import SqueezeExcitation
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class Heterogeneity():
    '''
    Main class for Multiscale Residual Spatiotemporal Vision Transformer (PixFormer)
    '''
    def __init__(self):
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.verbose      = True        # print progress
        self.return_data  = False       # return data
        self.folder       = 'Fdataset'  # dataset directory (F=fluvial, G=gaussian)
        self.lr           = 1e-3        # learning rate
        self.weight_decay = 1e-5        # weight decay for learning rate
        self.mse_weight   = 1.0         # Combined loss MSE weight
        self.ssim_weight  = 1.0         # Combined loss SSIM weight
        self.train_perc   = 0.75        # training set split percentage (of total)
        self.valid_perc   = 0.10        # validation set split percentage (of total)
        self.batch_size   = 32          # batch size
        self.num_epochs   = 100         # number of epochs
        self.check_torch_gpu()          # check if torch is built with GPU support

    def check_torch_gpu(self):
        '''
        Check if Torch is successfully built with GPU support
        '''
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        if self.verbose:
            print('\n'+'-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch version: {} | Torch Built with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('-'*60+'\n')
        return None
        
    def count_params(self, model):
        ### Count the total number of trainable parameters in the neural network
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def make_dataloaders(self, shuffle_valid=False, shuffle_test=False):
        '''
        Make the Train/Valid/Test dataloaders from custom Dataset and DataLoader classes
        '''
        print('-'*60+'\n'+'-------------- DATA LOADING AND PREPROCESSING --------------') if self.verbose else None
        file_names = os.listdir(self.folder)                              # list all files in directory
        file_paths = [os.path.join(self.folder, file_name) for file_name in file_names]
        transform  = TVTransform.Compose([                                # define image transformations
            TVTransform.ToImage(),                                        # convert to PIL image
            TVTransform.ToDtype(torch.float32),                           # convert to float32
            TVTransform.Resize(size=(64,64), antialias=True),             # resize to 64x64
            TVTransform.Normalize(mean=[0], std=[1]),                     # normalize by mean and std
            TVTransform.RandomHorizontalFlip(),                           # random horizontal flip
            TVTransform.RandomVerticalFlip()                              # random vertical flip
            ])
        dataset    = MyDataset(file_paths, transform, norm_type='MinMax') # create custom Dataset (with Transform and Normalization)
        train_size = int(self.train_perc * len(dataset))                  # define training set size
        valid_size = int(self.valid_perc * len(dataset))                  # define validation set size
        test_size  = len(dataset) - (train_size + valid_size)             # define testing set size
        if self.verbose:
            print('Total: {} - Training: {} | Validation: {} | Testing: {}'.format(len(dataset), train_size, valid_size, test_size))
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
        self.train_dataloader = MyDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,          mode='train')
        self.valid_dataloader = MyDataLoader(valid_dataset, batch_size=self.batch_size, shuffle=shuffle_valid, mode='valid')
        self.test_dataloader  = MyDataLoader(test_dataset,  batch_size=self.batch_size, shuffle=shuffle_test,  mode='test')
        if self.verbose:
            print('X single sample shape: {}'.format(self.train_dataloader.dataset[0][0].shape))
            print('y single sample shape: {}'.format(self.train_dataloader.dataset[0][1].shape))
        print(' '*24+'... done ...'+' '*24+'\n'+'-'*60+'\n')  if self.verbose else None
        if self.return_data:
            return self.train_dataloader, self.valid_dataloader, self.test_dataloader

    def trainer(self, optimizer=None, criterion=None):
        '''
        Subroutine for training the model
        '''
        print('-'*60+'\n'+'---------------------- MODEL TRAINING ----------------------') if self.verbose else None
        self.model = PixFormer().to(self.device) # instantiate model to device
        if optimizer is None:
            # AdamW optimizer with weight decay
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if criterion is None:
            # Custom loss function: L = a*MSE + b*(1-SSIM)
            criterion = CustomLoss(mse_weight=self.mse_weight, ssim_weight=self.ssim_weight).to(self.device)
        if self.verbose:
            print('Total number of trainable parameters: {:,}'.format(self.count_params(self.model)))
        train_loss, valid_loss, train_ssim, valid_ssim = [], [], [], [] # instanstiate empty lists for metrics
        best_val_loss, best_model = float('inf'), None                  # instantiate best model variables
        time0 = time.time()                                             # start overall timer
        for epoch in range(self.num_epochs):
            start_time = time.time()                                    # start epoch timer
            # Training
            self.model.train()                                          # set model to trainable mode
            epoch_loss, epoch_ssim = 0.0, 0.0                           # instantiate empty lists for storing training metrics
            for i, (x,y) in enumerate(self.train_dataloader):           # iterate over training batches
                x, y = x.to(self.device), y.to(self.device)             # move data to device
                optimizer.zero_grad()                                   # zero out gradients
                y_pred = self.model(x)                                  # forward pass
                loss = criterion(y_pred, y)                             # compute loss
                loss.backward()                                         # backpropagation
                optimizer.step()                                        # update weights
                epoch_loss += loss.item()                               # accumulate loss
                epoch_ssim += 1.0 - criterion.ssim(y_pred, y)           # accumulate SSIM
            train_loss.append(epoch_loss/(i+1))                         # append average loss
            train_ssim.append(epoch_ssim/(i+1))                         # append average SSIM
            # Validation
            self.model.eval()                                           # set model to evaluation mode
            epoch_loss, epoch_ssim = 0.0, 0.0                           # instantiate empty lists for storing validation metrics
            with torch.no_grad():                                       # disable gradient calculation
                for i, (x,y) in enumerate(self.valid_dataloader):       # iterate over validation batches
                    x, y = x.to(self.device), y.to(self.device)         # move data to device
                    y_pred = self.model(x)                              # forward pass
                    loss = criterion(y_pred, y)                         # compute loss
                    epoch_loss += loss.item()                           # accumulate loss
                    epoch_ssim += 1.0 - criterion.ssim(y_pred, y)       # accumulate SSIM
            valid_loss.append(epoch_loss/(i+1))                         # append average loss
            valid_ssim.append(epoch_ssim/(i+1))                         # append average SSIM
            # Save best model and losses
            if valid_loss[-1] < best_val_loss:                          # if validation loss is lower than previous best
                best_val_loss = valid_loss[-1]                          # update best validation loss
                best_model = self.model.state_dict()                    # update best model
            end_time = time.time() - start_time                         # end epoch timer
            if self.verbose:
                print('Epoch: {}/{} | Train loss: {:.4f} | Val loss: {:.4f} | Train SSIM: {:.4f} | Val SSIM: {:.4f} | Time elapsed: {:.2f} sec'.format(epoch+1, self.num_epochs, train_loss[-1], epoch_loss[-1], train_ssim[-1], epoch_ssim[-1], end_time))
        print('-'*60+'\n','Total training time: {:.2f} min'.format((time.time()-time0)/60), '\n'+'-'*60+'\n') if self.verbose else None
        print(' '*24+'... done ...'+' '*24+'\n'+'-'*60+'\n') if self.verbose else None
        self.losses = [train_loss, valid_loss, train_ssim, valid_ssim]
        torch.save(self.model, 'PixFormer_model.pt')
        return self.model, best_model, self.losses if self.return_data else None

    def tester(self, model=None, criterion=None):
        '''
        Subroutine for testing the model
        '''
        print('-'*60+'\n'+'---------------------- MODEL TESTING ----------------------') if self.verbose else None
        if model is None:
            model = torch.load('PixFormer_model.pt')                    # load best model
        if criterion is None:
            criterion = CustomLoss(mse_weight=self.mse_weight, ssim_weight=self.ssim_weight).to(self.device)
        model.eval()                                                    # set model to evaluation mode
        test_loss, test_ssim = 0.0, 0.0                                 # instantiate empty lists for storing testing metrics
        time0 = time.time()                                             # start overall timer
        with torch.no_grad():                                           # disable gradient calculation
            for i, (x,y) in enumerate(self.test_dataloader):            # iterate over testing batches
                x, y = x.to(self.device), y.to(self.device)             # move data to device
                y_pred = model(x)                                       # forward pass
                loss = criterion(y_pred, y)                             # compute loss
                test_loss += loss.item()                                # accumulate loss
                test_ssim += 1.0 - criterion.ssim(y_pred, y)            # accumulate SSIM
        test_loss /= (i+1)                                              # average loss
        test_ssim /= (i+1)                                              # average SSIM
        print('Test loss: {:.4f} | Test SSIM: {:.4f}'.format(test_loss, test_ssim)) if self.verbose else None
        print('-'*60+'\n','Total testing time: {:.2f} min'.format((time.time()-time0)/60), '\n'+'-'*60+'\n') if self.verbose else None
        print(' '*24+'... done ...'+' '*24+'\n'+'-'*60+'\n') if self.verbose else None
        return test_loss, test_ssim if self.return_data else None
    
    def plot_losses(self, losses=None):
        '''
        Plot training and validation losses
        '''
        if losses is None:
            losses = self.losses
        train_loss, valid_loss, train_ssim, valid_ssim = losses
        fig, ax = plt.subplots(1, 2, figsize=(12,4))
        ax1, ax2 = ax.flatten()
        ax1.plot(train_loss, label='Train loss'); ax1.plot(valid_loss, label='Valid loss')
        ax1.set_xlabel('Epochs', weight='bold');  ax1.set_ylabel('Loss', weight='bold')
        ax1.legend(facecolor='lightgray', edgecolor='k', fancybox=False)
        ax2.plot(train_ssim, label='Train SSIM'); ax2.plot(valid_ssim, label='Valid SSIM')
        ax2.set_xlabel('Epochs', weight='bold');  ax2.set_ylabel('SSIM', weight='bold')
        ax2.legend(facecolor='lightgray', edgecolor='k', fancybox=False)
        plt.tight_layout(); plt.savefig('losses.png', dpi=600, bbox_inches='tight')
        plt.show() if self.verbose else None  
        return None

class MyDataset(Dataset):
    '''
    Generate a custom dataset from .npz files
    (x) porosity, permeability, timesteps
    (y) pressure, saturation
    '''
    def __init__(self, file_paths, transform=None, norm_type:str='MinMax'):
        self.file_paths = file_paths
        self.transform  = transform
        self.tsteps     = 60
        self.x_channels = 3
        self.y_channels = 2
        self.orig_img   = 256
        self.half_img   = 64
        self.norm_type  = norm_type
        self.norm       = lambda x: self.normalize(x)

    def normalize(self, x):
        x_norm = np.zeros_like(x)
        error_msg = 'Invalid normalization scheme: {} | Select ["None", "MinMax", "ExtMinMax", "Standard"]'.format(self.norm_type)
        for i in range(x.shape[1]):
            if self.norm_type == 'MinMax':
                x_norm[:,i] = (x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min())
            elif self.norm_type == 'Standard':
                x_norm[:,i] = (x[:,i] - x[:,i].mean()) / (x[:,i].std())
            elif self.norm_type == 'ExtMinMax':
                x_norm[:,i] = (x[:,i] - x[:,i].min()) / (x[:,i].max() - x[:,i].min()) * 2 - 1
            elif self.norm_type == 'None':
                x_norm = x
            else:
                raise ValueError(error_msg)
        return x_norm

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        data   = np.load(self.file_paths[idx])
        poro   = np.tile(data['poro'], (self.tsteps, 1, 1, 1))
        perm   = np.tile(np.log10(data['perm']), (self.tsteps, 1, 1, 1))
        tstep  = np.tile(np.arange(1, self.tsteps+1).reshape(self.tsteps, 1, 1, 1), (1, 1, self.orig_img, self.orig_img))
        pres   = np.expand_dims(data['pres'], 1)
        sat    = np.expand_dims(data['sat'], 1)
        X_data = np.concatenate([poro, perm, tstep], axis=1).reshape(-1, self.x_channels, self.orig_img, self.orig_img)
        y_data = np.concatenate([pres, sat], axis=1).reshape(-1, self.y_channels, self.orig_img, self.orig_img)
        if self.transform:
            X_data_t = np.zeros((len(X_data), self.x_channels, self.half_img, self.half_img))
            y_data_t = np.zeros((len(y_data), self.y_channels, self.half_img, self.half_img))
            for i in range(len(X_data)):
                X_data_t[i] = self.transform(X_data[i].T)
                y_data_t[i] = self.transform(y_data[i].T)
            X_data, y_data = X_data_t, y_data_t
        x, y = torch.Tensor(self.norm(X_data)), torch.Tensor(self.norm(y_data))
        return x, y
    
class MyDataLoader(DataLoader):
    '''
    Generate a custom dataloader for dataset
    (train): x,y at timesteps 0-40
    (valid): x,y at timesteps 40-50
    (test):  x,y at timesteps 50-60
    '''
    def __init__(self, *args, mode:str=None, **kwargs):
        super(MyDataLoader, self).__init__(*args, **kwargs)
        self.mode = mode

    def __iter__(self):
        for batch in super(MyDataLoader, self).__iter__():
            X_data, y_data = batch          # loads a batch of data with shate (b, t, c, h, w)
            if self.mode == 'train':        # _____TRAINING_____
                X_data = X_data[:, :40]     # x at timesteps 0-40
                y_data = y_data[:, :40]     # y at timesteps 0-40
            elif self.mode == 'valid':      # _____VALIDATION_____
                X_data = X_data[:, 40:50]   # x at timesteps 40-50
                y_data = y_data[:, 40:50]   # y at timesteps 40-50
            elif self.mode == 'test':       # ______TESTING______
                X_data = X_data[:, 50:]     # x at timesteps 50-60
                y_data = y_data[:, 50:]     # y at timesteps 50-60
            else:
                raise ValueError('Invalid mode: {} | select between "train", "valid" or "test"'.format(self.mode))
            X_data = X_data[:, ::X_data.shape[1]//10]
            y_data = y_data[:, ::y_data.shape[1]//10]
            X_data = X_data.reshape(-1, X_data.size(2), X_data.size(3), X_data.size(4)) # reshape to (b*t, c, h, w)
            y_data = y_data.reshape(-1, y_data.size(2), y_data.size(3), y_data.size(4)) # reshape to (b*t, c, h, w)
            yield X_data, y_data

################################### MODEL CLASSES ###################################
class CustomLoss(nn.Module):
    '''
    Define custom loss function: L = a*MSE + b*(1-SSIM)
    '''
    def __init__(self, mse_weight=1.0, ssim_weight=1.0):
        super(CustomLoss, self).__init__()
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mse_weight  = mse_weight                                              # weights for MSE
        self.ssim_weight = ssim_weight                                             # weights for SSIM
        self.mse_loss    = nn.MSELoss()                                            # Mean Squared Error
        self.ssim_loss   = SSIM().to(self.device)                                  # Structural Similarity Index Measure
    def forward(self, pred, target):
        mse_loss   = self.mse_loss(pred, target)                                   # mse loss
        ssim_loss  = self.ssim_loss(pred, target)                                  # ssim loss
        total_loss = self.mse_weight * mse_loss + self.ssim_weight * (1-ssim_loss) # combined loss
        return total_loss
    
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
    def __init__(self, embed_dim, max_seq_len=256):
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
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim=256):
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
    def __init__(self, image_size=64, latent_size=16, in_channels=3, patch_size=8, projection_dim=64, embed_dim=128, num_heads=8, num_layers=4):
        super(ViTencoder, self).__init__()
        self.patch_embedding     = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)   # patch embedding
        self.positional_encoding = PositionalEncoding(embed_dim)                                    # positional encoding
        self.transformer_blocks  = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads) for _ in range(num_layers)])              # N Transfromer blocks
        self.global_avg_pooling  = nn.AdaptiveAvgPool1d(1)                                          # global average pooling
        self.fc = nn.Linear(embed_dim, projection_dim*latent_size*latent_size)                      # fully connected layer
    def forward(self, x):
        x = self.patch_embedding(x)                                                                 # patch embedding
        x = self.positional_encoding(x)                                                             # positional encoding
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)                                                                # transformer block(s)
        x = self.global_avg_pooling(x.transpose(1, 2))                                              # global average pooling
        x = x.squeeze(2)                                                                            # squeeze output
        x = self.fc(x)                                                                              # activate outputs
        return x
    
class MultiScaleResidual(nn.Module):
    '''
    Multiscale Residual concatenation block
    '''
    def __init__(self, image_size=64):
        super(MultiScaleResidual, self).__init__()
        self.image_size = image_size                                     # original image size
    def forward(self, x):
        _, _, h, w = x.shape                                             # get image dimensions
        scale = [self.image_size//h, self.image_size//w]                 # get scale factors
        size  = (self.image_size//scale[0], self.image_size//scale[1])   # get upscaled image size
        x_ups = TVTransform.Resize(size, antialias=True)(x)              # resize original image to upscaled size
        x = torch.cat([x, x_ups], dim=1)                                 # concatenate image and upscale image
        return x

class PixFormer(nn.Module):
    '''
    PixFormer model: 
    (1) Vision Transformer encoder
    (2) Multiscale Residual Spatiotemporal decoder
    '''
    def __init__(self, projection_dim=256, latent_size=8):
        super(PixFormer, self).__init__()
        self.projection_dim = projection_dim
        self.latent_size    = latent_size
        self.encoder = ViTencoder(latent_size=latent_size, projection_dim=projection_dim)
        self.layers = nn.Sequential(
            self._conv_block(projection_dim, projection_dim//2),                 # first decoder layer
            self._conv_block(projection_dim//2, projection_dim//4),              # second decoder layer
            self._conv_block(projection_dim//4, projection_dim//8))              # third decoder layer
        self.out = nn.Conv2d(projection_dim//8, 2, kernel_size=3, padding=1)     # output layer

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),      # convolve inputs
            SqueezeExcitation(out_channels, out_channels//4),                    # Squeeze-and-Excite for multichannel feature maps
            nn.InstanceNorm2d(out_channels),                                     # Normalize by instance
            nn.PReLU(),                                                          # Parametric ReLU activation
            MultiScaleResidual(),                                                # Multi-scale residual concatenation
            nn.Upsample(scale_factor=2),                                         # Upsample by 2x
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1))   # convolve concatenated features
    
    def forward(self, x):
        x = self.encoder(x)                                                       # encode inputs: z = Enc(x)
        x = x.view(-1, self.projection_dim, self.latent_size, self.latent_size)   # reshape z: vector -> feature maps
        for layer in self.layers:
            x = layer(x)                                                          # decode inputs: y = Dec(z)
        x_output = self.out(x)                                                    # output layer
        return x_output    

############################## MAIN ##############################
if __name__ == '__main__': 
    # Run main routine if main.py called directly
    hete = Heterogeneity()
    hete.make_dataloaders()
    hete.trainer()
    hete.tester()
    hete.plot_losses()
############################## END ##############################