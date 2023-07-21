import os
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchvision.utils import save_image
from torchsummary import summary
from torchviz import make_dot
import torchio as tio

class L2normaliation(nn.Module):
    def __init__(self, dim=1, eps=1e-12):
        super(L2normaliation, self).__init__()
        self.dim = dim
        self.eps = eps
    def fowrard(self, x):
        return F.normalize(x, p=2, dim=self.dim, eps=self.eps)
     
class NumpyDataset_fromFolder(Dataset):
    def __init__(self, folder_X, folder_y):
        self.folder_X = folder_X
        self.folder_y = folder_y
        self.X_filenames = sorted(os.listdir(folder_X))
        self.y_filenames = sorted(os.listdir(folder_y))
    def __len__(self):
        return len(self.X_filenames)
    def __getitem__(self, index):
        X_path = os.path.join(self.folder_X, self.X_filenames[index])
        y_path = os.path.join(self.folder_y, self.y_filenames[index])
        X, y = np.load(X_path), np.load(y_path)
        return X, y
    
class NumpyDataset_from_array(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        img_x = self.X[index]
        img_y = self.y[index]        
        return img_x, img_y
    
class CustomLoss(nn.Module):
    def __init__(self, mse_weight=1.0, ssim_weight=1.0, todevice=True):
        super(CustomLoss, self).__init__()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.mse_loss = nn.MSELoss()
        if todevice:
          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          self.ssim = SSIM().to(device)
        else:
          self.ssim = SSIM()
    def forward(self, pred, target):
        mse_loss = self.mse_loss(pred, target)
        ssim_loss = 1.0 - self.ssim(pred, target)
        total_loss = self.mse_weight * mse_loss + self.ssim_weight * ssim_loss
        return total_loss
    
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embedding_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.conv = nn.Conv2d(in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size)
        self.num_patches = (64 // patch_size) ** 2
    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4):
        super(SwinTransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim))
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.window_size = window_size
    def forward(self, x):
        x = x.permute(1, 0, 2)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 0, 2)
        x = x + x
        x = self.norm1(x)
        x = self.mlp(x)
        x = x + x
        x = self.norm2(x)
        return x

class SwinTransformer(nn.Module):
    def __init__(self, in_channels=4, output_channels=2, embedding_dim=256, patch_size=4, num_heads=8, num_layers=6, slope=0.2):
        super(SwinTransformer, self).__init__()
        ch = output_channels
        self.output_channels = output_channels
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.num_patches = self.patch_embed.num_patches
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches + 1, embedding_dim))
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(embedding_dim, num_heads, window_size=2 ** (i + 1)) for i in range(num_layers)])
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, output_channels))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(output_channels, ch*64, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch*64),
            nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(ch*64, ch*32, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch*32),
            nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(ch*32, ch*16, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch*16),
            nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(ch*16, ch*8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch*8),
            nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(ch*8, ch*4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(ch*4),
            nn.LeakyReLU(slope, inplace=True),
            nn.ConvTranspose2d(ch*4, ch, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid())
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_encoding[:, :self.num_patches]
        for swin_block in self.swin_blocks:
            x = swin_block(x)
        x = x.mean(dim=1)
        x = self.mlp_head(x)
        x = x.view(-1, self.output_channels, 1, 1)
        x = self.decoder(x)
        x = x + F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return x 
   
class VisionTransformer(nn.Module):
    def __init__(self, in_channels=4, output_channels=2, embedding_dim=256, patch_size=4, num_heads=8, num_layers=6):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embedding_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.patch_embed.num_patches + 1, embedding_dim))
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers)
        self.decoder = nn.Linear(embedding_dim * self.patch_embed.num_patches, output_channels * 64 * 64)
    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_encoding[:, :x.size(1)]
        x = self.transformer_encoder(x)
        x = x.flatten(1, 2)
        x = self.decoder(x)
        x = x.view(x.size(0), -1, 64, 64)
        return x