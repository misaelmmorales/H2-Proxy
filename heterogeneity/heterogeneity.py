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
    def __init__(self):
        self.verbose      = True
        self.return_data  = False
        self.folder       = 'Fdataset'
        self.lr           = 1e-3
        self.weight_decay = 1e-5
        self.mse_weight   = 1.0
        self.ssim_weight  = 1.0
        self.train_perc   = 0.75
        self.valid_perc   = 0.15
        self.batch_size   = 32
        self.num_epochs   = 100
        self.check_torch_gpu()

    def check_torch_gpu(self):
        torch_version, cuda_avail = torch.__version__, torch.cuda.is_available()
        self.device = torch.device('cuda' if cuda_avail else 'cpu')
        count, name = torch.cuda.device_count(), torch.cuda.get_device_name()
        if self.verbose:
            print('\n'+'-'*60)
            print('----------------------- VERSION INFO -----------------------')
            print('Torch Version {} | Torch Build with CUDA? {}'.format(torch_version, cuda_avail))
            print('# Device(s) available: {}, Name(s): {}'.format(count, name))
            print('-'*60+'\n')
        return None
    
    def count_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)