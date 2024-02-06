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
        self.verbose      = True                # print progress
        self.return_data  = False               # return data
        self.folder       = 'train_dataF_64x64' # dataset directory (F=fluvial, G=gaussian)
        self.lr           = 1e-3                # learning rate
        self.weight_decay = 1e-5                # weight decay for learning rate
        self.mse_weight   = 1.0                 # Combined loss MSE weight
        self.ssim_weight  = 1.0                 # Combined loss SSIM weight
        self.train_perc   = 0.10                # training set split percentage (of total)
        self.valid_perc   = 0.05                # validation set split percentage (of total)
        self.batch_size   = 32                  # batch size
        self.num_epochs   = 100                 # number of epochs
        self.check_torch_gpu()                  # check if torch is built with GPU support

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
        dataset = CustomDataset(self.folder)
        train_size = int(self.train_perc * len(dataset))
        valid_size = int(self.valid_perc * len(dataset))
        test_size  = len(dataset) - train_size - valid_size
        train_data, valid_data, test_data = random_split(dataset, [train_size, valid_size, test_size])
        self.train_loader = CustomDataloader(train_data, mode='train', batch_size=self.batch_size, shuffle=True)
        self.valid_loader = CustomDataloader(valid_data, mode='valid', batch_size=self.batch_size, shuffle=True)
        self.test_loader  = CustomDataloader(test_data,  mode='test',  batch_size=self.batch_size, shuffle=True)
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
        self.model = H2ViTnet().to(self.device)
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
            if self.verbose:
                print('Epoch: {}/{} | Train loss: {:.4f} | Val loss: {:.4f} | Time elapsed: {:.2f} sec'.format(epoch+1, self.num_epochs, train_loss[-1], valid_loss[-1], end_time))
        print('-'*60+'\n','Total training time: {:.2f} min'.format((time.time()-time0)/60), '\n'+'-'*60+'\n') if self.verbose else None
        print(' '*24+'... done ...'+' '*24+'\n'+'-'*60+'\n') if self.verbose else None
        self.losses = [train_loss, valid_loss]
        torch.save(self.model.state_dict(), 'h2vit_model.pth')
        return self.model, self.losses if self.return_data else None
    
    def plot_losses(self, losses):
        '''
        Plot the training and validation losses
        '''
        train_loss, valid_loss = losses
        plt.figure(figsize=(10,5))
        plt.plot(train_loss, label='Train loss')
        plt.plot(valid_loss, label='Valid loss')
        plt.xlabel('Epochs', weight='bold'); plt.ylabel('Loss', weight='bold')
        plt.title('Training and Validation Losses', weight='bold')
        plt.legend(facecolor='lightgrey', edgecolor='k', fancybox=False)
        plt.tight_layout()
        plt.savefig('losses.png', dpi=600, bbox_inches='tight')
        plt.show() if self.verbose else None
        return None

#####################################################################################
########################### DATA LOADING AND PREPROCESSING ##########################
#####################################################################################
class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = os.listdir(data_folder)
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_path = os.path.join(self.data_folder, self.file_list[idx])
        data = np.load(file_path)
        return torch.Tensor(data['X_data']), torch.Tensor(data['y_data'] )
    
class CustomDataloader(DataLoader):
    def __init__(self, *args, mode:str=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
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
            yield X_data, y_data    

#####################################################################################
################################### MODEL CLASSES ###################################
#####################################################################################
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
    def __init__(self, image_size=256, in_channels=3,
                 num_heads=8, num_layers=2, latent_size=32, patch_size=8, projection_dim=64, embed_dim=16,
                 max_seq_len=1024, mlp_hidden_dim=64):
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
    def __init__(self, projection_dim=64, latent_size=32):
        super(H2ViTnet, self).__init__()
        self.projection_dim = projection_dim
        self.latent_size    = latent_size

        self.encoder = ViTencoder(projection_dim=projection_dim, latent_size=latent_size)
        self.layers = nn.Sequential(
            self._conv_block(projection_dim, projection_dim//2),
            self._conv_block(projection_dim//2, projection_dim//4),
            self._conv_block(projection_dim//4, projection_dim//8),
        )
        self.out = nn.Conv2d(projection_dim//8, 2, kernel_size=3, padding=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            SqueezeExcitation(out_channels, out_channels//4),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(),
            nn.Upsample(scale_factor=2),
        )
    
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
    h.plot_losses()
    print(' '*24+'... done ...'+' '*24+'\n'+'-'*60+'\n') if h.verbose else None

#####################################################################################
#################################### END OF FILE ####################################
#####################################################################################