# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:46:47 2022

@author: tripa
"""

'''
# %% Run below cell when using colab

# Mount google-drive
from google.colab import drive
drive.mount('/content/gdrive')

# Copy necessary files to the current environment
!cp gdrive/MyDrive/Colab/Colorization/utils.py .
!cp gdrive/MyDrive/Colab/Colorization/generator.py .
!cp gdrive/MyDrive/Colab/Colorization/discriminator.py .

# Extract the VOC dataset in the current environment
!unzip gdrive/MyDrive/Colab/VOCdevkit.zip
'''

import os
import time
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from utils import load_transformed_batch, init_weights
from generator import UNet
from discriminator import PatchGAN

import torch
from torch import optim
from torchvision import transforms

# runtime = 'colab' or 'local'
runtime = 'local'

if runtime == 'local':
    res_dir = os.path.join(os.getcwd(), 'Colorization')
else:
    res_dir = os.path.join(os.getcwd(), 'gdrive', 'MyDrive', 'Colab', 'Colorization')
    
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

# %%

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Root directory for VOC data
voc_root = os.path.join(os.getcwd(), 'VOCdevkit', 'VOC2007', 'JPEGImages')
data_files = os.listdir(voc_root)

train_files = data_files[:3000]  # Hardcoded for now, split into 80:20 later


# %%

# Check some sample images

random_files = np.random.choice(train_files, size=16)
random_samples = [os.path.join(voc_root, x) for x in random_files]

_, axes = plt.subplots(4, 4, figsize=(10, 10))
for ax, img_path in zip(axes.flatten(), random_samples):
    ax.imshow(Image.open(img_path))
    ax.axis("off")

# L, ab, target = train_dataloader.__getitem__(0)
# L = (L + 1.) * 50.
# ab = ab * 110
# img_LAB = torch.cat([L, ab], dim=0).detach().cpu().numpy()
# #img_LAB = np.transpose(img_LAB, (1, 2, 0))
# #img_RGB = lab2rgb(img_LAB)
# #plt.imshow(img_RGB, cmap='gray')
# #plt.show()

# plt.imshow(img_LAB[0])
# plt.show()

# %%

# Transformations for the training data
train_transforms = transforms.Compose([transforms.Resize((256, 256), Image.BICUBIC),
                                       transforms.RandomHorizontalFlip()])  # for data augmentation

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for
n_epochs = 15

# Set-up hyperparameter lambda for L1 term
lmbda = 100.

# # Create real_label and fake_labels
# register_buffer('real_label', torch.tensor(1.0))
# register_buffer('fake_label', torch.tensor(0.0))

# Create generator object and initialize weights (normally)
generator = UNet(in_channels=1, out_channels=2, n_filters=64)
generator = init_weights(generator)
# nn.init.normal_(generator.weight.data, mean=0.0, std=0.02)  # Does it initialize all layer weights? Re-check!!
generator.to(device)

# Create discriminator object and initialize weights (normally)
discriminator = PatchGAN(in_channels=3)
discriminator = init_weights(discriminator)
# nn.init.normal_(discriminator.weight.data, mean=0.0, std=0.02)  # Does it initialize all layer weights? Re-check!!
discriminator.to(device)

# Set-up optimizer and scheduler
generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# Set-up loss function
l1_criterion = torch.nn.L1Loss()
criterion = torch.nn.BCEWithLogitsLoss()

# Set-up labels for real and fake predictions
real_label = torch.tensor(1.0)
fake_label = torch.tensor(0.0)

# Calculate the number of batches
n_batches = int(len(train_files)/batch_size)

for epoch in range(n_epochs):
    
    # Variable to record time taken in each epoch
    since = time.time()
    
    print('Epoch {}/{}'.format(epoch, n_epochs - 1))
    print('-' * 10)

    running_generator_loss_adversarial = 0.0
    running_generator_loss_l1 = 0.0
    running_generator_loss_total = 0.0
    running_discriminator_loss_real = 0.0
    running_discriminator_loss_fake = 0.0
    running_discriminator_loss_total = 0.0
    
    # Iterate over all the batches
    for j in range(n_batches):
            
        print('Batch {}/{}'.format(j+1, n_batches+1))
            
        # Get the train data and labels for the current batch
        batch_files = train_files[j*batch_size:(j+1)*batch_size]
        L, ab = load_transformed_batch(voc_root, batch_files, train_transforms)
        
        # Put the data to the device
        L, ab = L.to(device), ab.to(device)
        
        # Train the discriminator
        discriminator.train()
        
        with torch.set_grad_enabled(True):
            
            #discriminator.set_requires_grad(True)
        
            # Make gradients zero before forward pass
            discriminator_optimizer.zero_grad()
            
            # Create a fake color image using the generator
            fake_color = generator(L)
            
            # Run fake examples through the discriminator
            fake_image = torch.cat([L, fake_color], dim=1)  # Make dim=0 when passing only one sample
            fake_preds = discriminator(fake_image.detach())
            discriminator_loss_fake = criterion(fake_preds, fake_label.expand_as(fake_preds).to(device))
            
            # Run real examples through the discriminator
            real_image = torch.cat([L, ab], dim=1)  # Make dim=0 when passing only one sample
            real_preds = discriminator(real_image)
            discriminator_loss_real = criterion(real_preds, real_label.expand_as(real_preds).to(device))
            
            # Total loss is the sum of both the losses
            discriminator_loss_total = (discriminator_loss_fake + discriminator_loss_real) * 0.5
            
            # backward + optimize
            discriminator_loss_total.backward()
            discriminator_optimizer.step()
        
        # Train the generator while keeping the discriminator weights constant
        generator.train()
        # discriminator.set_requires_grad(False)
        
        # Make gradients zero before forward pass
        generator_optimizer.zero_grad()
        
        # Calculate the prediction using discriminator
        fake_preds = discriminator(fake_image)
        
        # Calculate adversarial loss for the generator
        generator_loss_adversarial = criterion(fake_preds, real_label.expand_as(real_preds).to(device))
        
        # Calculate L1 loss for the generator (lambda * L1_loss)
        generator_loss_l1 = l1_criterion(fake_color, ab) * lmbda
        
        # Total loss is the sum of both the losses
        generator_loss_total = generator_loss_adversarial + generator_loss_l1
        
        # backward + optimize
        generator_loss_total.backward()
        generator_optimizer.step()
        
        # running_discriminator_loss_fake += discriminator_loss_fake.item() * batch_size
        # running_discriminator_loss_real += discriminator_loss_real.item() * batch_size
        running_discriminator_loss_total += discriminator_loss_total.item() * batch_size
        # running_generator_loss_adversarial += generator_loss_adversarial.item() * batch_size
        # running_generator_loss_l1 += generator_loss_l1.item() * batch_size
        running_generator_loss_total += generator_loss_total.item() * batch_size

    # epoch_discriminator_loss_fake = running_discriminator_loss_fake / (n_batches*batch_size)
    # epoch_discriminator_loss_real = running_discriminator_loss_real / (n_batches*batch_size)
    epoch_discriminator_loss_total = running_discriminator_loss_total / (n_batches*batch_size)
    # epoch_generator_loss_adversarial = running_generator_loss_adversarial / (n_batches*batch_size)
    # epoch_generator_loss_l1 = running_generator_loss_l1 / (n_batches*batch_size)
    epoch_generator_loss_total = running_generator_loss_total / (n_batches*batch_size)
    
    print('Discriminator Loss: {:.4f} Generator Loss: {:.4f}'.format(epoch_discriminator_loss_total, epoch_generator_loss_total))
        
    time_elapsed = time.time() - since
    print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    # The below statements are run after every epoch for saving the model
    # Save the generator and discriminator model
    if epoch % 5 == 0 or epoch == n_epochs-1:
        torch.save(generator.state_dict(), os.path.join(res_dir, 'generator_' + str(epoch) + '.pth'))
        torch.save(discriminator.state_dict(), os.path.join(res_dir, 'discriminator_' + str(epoch) + '.pth'))


# %%
