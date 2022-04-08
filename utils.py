# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:36:07 2022

@author: tripa
"""

import os
from PIL import Image
import numpy as np
from skimage.color import rgb2lab

import torch
from torch import nn
from torchvision import transforms

def init_weights(model):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and 'Conv' in classname:
            nn.init.normal_(m.weight.data, mean=0.0, std=0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif 'BatchNorm2d' in classname:
            nn.init.normal_(m.weight.data, 1., 0.02)
            nn.init.constant_(m.bias.data, 0.)
    model.apply(init_func)
    print("model initialized")
    return model

def load_transformed_batch(data_dir, batch_files, data_transforms):

    L_channels = []
    a_and_b_channels = []
    
    # Enumerate over all the files in the batch_files list
    for i, x in enumerate(batch_files):
    
        # Open image as PIL and apply transformations
        image = Image.open(os.path.join(data_dir, x)).convert("RGB")
        image = data_transforms(image)
        
        # Convert the image from RGB to LAB format
        LAB_image = rgb2lab(np.array(image)).astype('float32')
        LAB_image = transforms.ToTensor()(LAB_image)
        
        L = LAB_image[[0], ...] / 50. - 1.  # Between -1 and 1
        ab = LAB_image[[1, 2], ...] / 110.  # Between -1 and 1
        
        # For the first iteration create a tensor
        if i == 0:
            L_channels = L.reshape(1, 1, 256, 256)
            a_and_b_channels = ab.reshape(1, 2, 256, 256)
        # For subsequent iterations concatenate tensors to the data
        else:
            L_channels = torch.cat([L_channels, L.reshape(1, 1, 256, 256)], axis=0)
            a_and_b_channels = torch.cat([a_and_b_channels, ab.reshape(1, 2, 256, 256)], axis=0)

    return L_channels, a_and_b_channels

# %%