import os, time, math
import numpy as np
import matplotlib.pyplot as plt
from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from torchvision.ops import SqueezeExcitation
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM

class Heterogeneity():
    def __init__(self):
        self.verbose = True
        self.folder  = 'Fdataset'
        
        self.device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr           = 1e-3
        self.weight_decay = 1e-5
        self.mse_weight   = 1.0
        self.ssim_weight  = 1.0

        self.train_perc = 0.75
        self.valid_perc = 0.10

        self.batch_size = 32
        self.num_epochs = 100

    def check_torch_gpu(self):
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        #py_version, conda_env_name = sys.version, sys.executable.split('\\')[-2]
        print('\n'+'-'*60)
        print('----------------------- VERSION INFO -----------------------')
        #print('Conda Environment: {} | Python version: {}'.format(conda_env_name, py_version))
        print('Torch version: {}'.format(torch_version))
        print('Torch build with CUDA? {}'.format(cuda_avail))
        print('# Device(s) available: {}, Name(s): {}'.format(count, name))
        print('-'*60+'\n')
        return None
        
    def count_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def make_dataloaders(self, shuffle_valid=False, shuffle_test=False):
        file_names = os.listdir(self.folder)
        file_paths = [os.path.join(self.folder, file_name) for file_name in file_names]
        dataset    = MyDataset(file_paths)
        train_size = int(self.train_perc * len(dataset))
        valid_size = int(self.valid_perc * len(dataset))
        test_size  = len(dataset) - (train_size + valid_size)
        train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_size, valid_size, test_size])
        self.train_dataloader = MyDataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  mode='train')
        self.valid_dataloader = MyDataLoader(valid_dataset, batch_size=self.batch_size, shuffle=shuffle_valid, mode='valid')
        self.test_dataloader  = MyDataLoader(test_dataset,  batch_size=self.batch_size, shuffle=shuffle_test,  mode='test')
        if self.verbose:
            return self.train_dataloader, self.valid_dataloader, self.test_dataloader

    def trainer(self, optimizer=None, criterion=None):
        self.model = PixFormer().to(self.device)
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if criterion is None:
            criterion = CustomLoss(mse_weight=self.mse_weight, ssim_weight=self.ssim_weight).to(self.device)
        print('-'*60+'\n'+'Total number of trainable parameters: {:,}'.format(self.count_params(self.model)))

        train_loss, valid_loss = [], []
        train_ssim, valid_ssim = [], []
        best_val_loss = float('inf')
        best_model = None
        time0 = time.time()

        for epoch in range(self.num_epochs):
            start_time = time.time()

            self.model.train()
            epoch_loss, epoch_ssim = 0.0, 0.0
            for i, (x,y) in enumerate(self.train_dataloader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                y_pred = self.model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                epoch_ssim += 1.0 - criterion.ssim(y_pred, y)
            train_loss.append(epoch_loss/(i+1))
            train_ssim.append(epoch_ssim/(i+1))

            self.model.eval()
            epoch_loss, epoch_ssim = 0.0, 0.0
            with torch.no_grad():
                for i, (x,y) in enumerate(self.valid_dataloader):
                    x, y = x.to(self.device), y.to(self.device)
                    y_pred = self.model(x)
                    loss = criterion(y_pred, y)
                    epoch_loss += loss.item()
                    epoch_ssim += 1.0 - criterion.ssim(y_pred, y)
            valid_loss.append(epoch_loss/(i+1))
            valid_ssim.append(epoch_ssim/(i+1))

            if valid_loss[-1] < best_val_loss:
                best_val_loss = valid_loss[-1]
                best_model = self.model.state_dict()
            
            end_time = time.time() - start_time
            print('Epoch: {}/{} | Train loss: {:.4f} | Val loss: {:.4f} | Train SSIM: {:.4f} | Val SSIM: {:.4f} | Time elapsed: {:.2f} sec'.format(epoch+1, epochs, train_loss[-1], val_loss[-1], train_ssim[-1], val_ssim[-1], end_time))
        print('-'*60+'\n','Total training time: {:.2f} min'.format((time.time()-time0)/60), '\n'+'-'*60+'\n')
        return self.model, best_model, train_loss, valid_loss, train_ssim, valid_ssim

class CustomLoss(nn.Module):
    def __init__(self, mse_weight=1.0, ssim_weight=1.0):
        super(CustomLoss, self).__init__()
        self.device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mse_weight  = mse_weight
        self.ssim_weight = ssim_weight
        self.mse_loss    = nn.MSELoss()
        self.ssim        = SSIM().to(self.device)

    def forward(self, pred, target):
        mse_loss   = self.mse_loss(pred, target)
        ssim_loss  = 1.0 - self.ssim(pred, target)
        total_loss = self.mse_weight * mse_loss + self.ssim_weight * ssim_loss
        return total_loss
    
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        patches = self.projection(x)
        patches = rearrange(patches, 'b c h w -> b (h w) c')
        return patches
    
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_seq_len=512):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        pos_enc  = torch.zeros((1, max_seq_len, embed_dim))
        pos_enc[0, :, 0::2] = torch.sin(position * div_term)
        pos_enc[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        return x + self.pos_enc[:, :x.size(1)].detach()
    
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key   = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]
        Q = self.query(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attention_weights, V)
        out = out.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.embed_dim)
        out = self.fc_out(out)
        return out
    
class MLPBlock(nn.Module):
    def __init__(self, embed_dim, mlp_hidden_dim):
        super(MLPBlock, self).__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, embed_dim)

    def forward(self, x):
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim=1024):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attention = MultiHeadAttention(embed_dim, num_heads)
        self.mlp_block = MLPBlock(embed_dim, mlp_hidden_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attention_output = self.self_attention(x, x, x)
        x = x + attention_output
        x = self.norm1(x)
        mlp_output = self.mlp_block(x)
        x = x + mlp_output
        x = self.norm2(x)
        return x
    
class ViTencoder(nn.Module):
    def __init__(self, image_size=256, latent_size=32, in_channels=3, patch_size=16, projection_dim=256, embed_dim=1024, num_heads=16, num_layers=8):
        super(ViTencoder, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.transformer_blocks = nn.ModuleList([TransformerEncoderBlock(embed_dim, num_heads) for _ in range(num_layers)])
        self.global_avg_pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(embed_dim, projection_dim*latent_size*latent_size)

    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        x = self.global_avg_pooling(x.transpose(1, 2))
        x = x.squeeze(2)
        x = self.fc(x)
        return x
    
class MultiScaleResidual(nn.Module):
    def __init__(self, image_size=256):
        super(MultiScaleResidual, self).__init__()
        self.image_size = image_size

    def forward(self, x):
        _, _, h, w = x.shape
        scale = [self.image_size//h, self.image_size//w]
        size  = (self.image_size//scale[0], self.image_size//scale[1])
        x_ups = transforms.Resize(size, antialias=True)(x)
        return torch.cat([x, x_ups], dim=1)
    
class PixFormer(nn.Module):
    def __init__(self, projection_dim=128, latent_size=32):
        super(PixFormer, self).__init__()
        self.projection_dim = projection_dim
        self.latent_size    = latent_size
        self.encoder = ViTencoder(latent_size=latent_size, projection_dim=projection_dim)
        self.layers = nn.Sequential(
            self._conv_block(projection_dim, projection_dim//2),
            self._conv_block(projection_dim//2, projection_dim//4),
            self._conv_block(projection_dim//4, projection_dim//8))
        self.out = nn.Conv2d(projection_dim//8, 1, kernel_size=3, padding=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            SqueezeExcitation(out_channels, out_channels//4),
            nn.InstanceNorm2d(out_channels),
            nn.PReLU(),
            MultiScaleResidual(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(out_channels*2, out_channels, kernel_size=3, padding=1))

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.projection_dim, self.latent_size, self.latent_size)
        for layer in self.layers:
            x = layer(x)
        x_output = self.out(x)
        return x_output
    
class MyDataset(Dataset):
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx])
        poro = np.tile(data['poro'], (60, 1, 1, 1))
        perm = np.tile(data['perm'], (60, 1, 1, 1))
        timesteps = np.tile(np.arange(1, 61).reshape(60, 1, 1, 1), (1, 1, 256, 256))
        pres = data['pres'].reshape(60, 1, 256, 256)
        sat = data['sat'].reshape(60, 1, 256, 256)
        X_data = np.concatenate([poro, perm, timesteps], axis=1).reshape(-1, 3, 256, 256)
        y_data = np.concatenate([pres, sat], axis=1).reshape(-1, 2, 256, 256)
        return torch.Tensor(X_data), torch.Tensor(y_data)
    
class MyDataLoader(DataLoader):
    def __init__(self, *args, mode:str=None, **kwargs):
        super(MyDataLoader, self).__init__(*args, **kwargs)
        self.mode = mode

    def __iter__(self):
        for batch in super(MyDataLoader, self).__iter__():
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
                raise ValueError('Invalid mode: {} | select between "train", "valid" or "test"'.format(self.mode))
            X_data = X_data.reshape(-1, X_data.size(2), X_data.size(3), X_data.size(4))
            y_data = y_data.reshape(-1, y_data.size(2), y_data.size(3), y_data.size(4))
            yield X_data, y_data

if __name__ == '__main__':
    hete = Heterogeneity()
    hete.check_torch_gpu()
    hete.make_dataloaders()
    hete.trainer()