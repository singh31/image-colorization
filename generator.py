# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 00:12:31 2022

@author: tripa
"""

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class UNetBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, downsample=True, use_dropout=False):
        super().__init__()
        
        if downsample:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            self.act = nn.LeakyReLU(0.2)
        else:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            self.act = nn.ReLU()
            
        self.norm = nn.BatchNorm2d(out_channels)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        
        if self.use_dropout:
            return self.dropout(self.act(self.norm(self.conv(x))))
        else:
            return self.act(self.norm(self.conv(x)))
        
        
class UNet(nn.Module):
    
    def __init__(self, in_channels=3, out_channels=3, n_filters=64):
        super().__init__()
        
        self.downsample_initial = nn.Sequential(nn.Conv2d(in_channels, n_filters, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2))
        
        self.down1 = UNetBlock(n_filters, n_filters*2, downsample=True,use_dropout=False)
        self.down2 = UNetBlock(n_filters*2, n_filters*4, downsample=True,use_dropout=False)
        self.down3 = UNetBlock(n_filters*4, n_filters*8, downsample=True,use_dropout=False)
        self.down4 = UNetBlock(n_filters*8, n_filters*8, downsample=True,use_dropout=False)
        self.down5 = UNetBlock(n_filters*8, n_filters*8, downsample=True,use_dropout=False)
        self.down6 = UNetBlock(n_filters*8, n_filters*8, downsample=True,use_dropout=False)
        
        self.downsample_inner = nn.Sequential(nn.Conv2d(n_filters*8, n_filters*8, kernel_size=4, stride=2, padding=1), nn.ReLU())
        self.upsample_inner = UNetBlock(n_filters*8, n_filters*8, downsample=False, use_dropout=True)
        
        self.up1 = UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True)
        self.up2 = UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=True)
        self.up3 = UNetBlock(n_filters*8*2, n_filters*8, downsample=False, use_dropout=False)
        self.up4 = UNetBlock(n_filters*8*2, n_filters*4, downsample=False, use_dropout=False)
        self.up5 = UNetBlock(n_filters*4*2, n_filters*2, downsample=False, use_dropout=False)
        self.up6 = UNetBlock(n_filters*2*2, n_filters, downsample=False, use_dropout=False)
        
        self.upsample_final = nn.Sequential(nn.ConvTranspose2d(n_filters*2, out_channels, kernel_size=4, stride=2, padding=1), nn.Tanh())
        
    def forward(self, x):
        
        d_initial = self.downsample_initial(x)
        d1 = self.down1(d_initial)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        
        d_inner = self.downsample_inner(d6)
        u_inner = self.upsample_inner(d_inner)
        
        u1 = self.up1(torch.cat([u_inner, d6], dim=1))
        u2 = self.up2(torch.cat([u1, d5], dim=1))
        u3 = self.up3(torch.cat([u2, d4], dim=1))
        u4 = self.up4(torch.cat([u3, d3], dim=1))
        u5 = self.up5(torch.cat([u4, d2], dim=1))
        u6 = self.up6(torch.cat([u5, d1], dim=1))
        
        u_final = self.upsample_final(torch.cat([u6, d_initial], dim=1))
        
        return u_final

# %%

###############################################################################
#                       Test the generator model (U-Net)                      #
###############################################################################

# Create a random noise vector of suitable shape
noise_vector = torch.randn((1, 3, 256, 256))

# Create U-Net model object with 3 input channels and 64 features
model = UNet(in_channels=3, out_channels=3, n_filters=64)

# Run model to get the output
output = model(noise_vector)

# Convert the output to numpy array
output = output.cpu().detach().numpy()

# Transpose the output array to make image-channels as the last dimensions
output = np.transpose(output[0], (1, 2, 0))

# Display the image
plt.imshow(output)
plt.show()
