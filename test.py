# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 10:30:40 2022

@author: tripa
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import lab2rgb

from generator import UNet
from utils import load_transformed_batch

import torch
from torchvision import transforms

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

def lab_to_rgb(L, ab):
    """
    Takes a batch of images
    """
    
    L = (L + 1.) * 50.
    ab = ab * 110.
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb_imgs = []
    for img in Lab:
        img_rgb = lab2rgb(img)
        rgb_imgs.append(img_rgb)
    return np.stack(rgb_imgs, axis=0)

# %%

# Root directory for VOC data
voc_root = os.path.join(os.getcwd(), 'VOCdevkit', 'VOC2007', 'JPEGImages')

# Transformations for testing the model (resize input image to suitable size for model input)
val_transforms = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC)])

# Create generator object and initialize weights (normally)
generator = UNet(in_channels=1, out_channels=2, n_filters=64)

# Load the trained model weights
if device.type == 'cpu':
    generator.load_state_dict(torch.load('generator_14.pth', map_location=torch.device('cpu')))
else:
    generator.load_state_dict(torch.load('generator_14.pth'))

generator.eval()  # Since we are using only for testing
generator.requires_grad_(False)

data_files = os.listdir(voc_root)
L, ab = load_transformed_batch(voc_root, data_files[7005:7010], val_transforms)
L, ab = L.to(device), ab.to(device)

fake_color = generator(L)
real_color = ab

fake_images = lab_to_rgb(L, fake_color)
real_images = lab_to_rgb(L, real_color)

# %%

fig = plt.figure(figsize=(15, 8))
for i in range(5):
    
    ax = plt.subplot(3, 5, i + 1)
    ax.imshow(L[i][0].cpu(), cmap='gray')
    ax.axis("off")
    ax = plt.subplot(3, 5, i + 1 + 5)
    ax.imshow(fake_images[i])
    ax.axis("off")
    ax = plt.subplot(3, 5, i + 1 + 10)
    ax.imshow(real_images[i])
    ax.axis("off")
plt.show()