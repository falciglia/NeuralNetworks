'''#############################################################'''
'''#################### Importing libraries ####################'''
'''#############################################################'''

import torch
import torch.utils
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn as nn
import torch.nn.functional as Functional
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.utils as utils

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import Callback

import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.stats import entropy
import math
import random
from skimage.transform import resize

import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import copy

import sys, os, time
import scipy.io as io
import scipy.stats as stats

import seaborn as sns
from PIL import Image

SEED = 2206

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Specify device
device = 'cuda' 

# Specify the batch size.
batch_size = 5

# Data folder
data_folder="/raid/home/falcigls/NeuralNetworks/Dataset/69_dataset/69dataset.mat"



'''#############################################################'''
'''################### Dataset & Data Module ###################'''
'''#############################################################'''

class MNIST69_Dataset(Dataset):

    def __init__(self,
                 batch_size,
                 mode,
                 data_folder):
        self.batch_size = batch_size
        self.mode = mode
        self.data_folder = data_folder

        self.dataset = self.open_dataset(self.data_folder, self.mode)

        self.transform = transforms.Compose([transforms.Resize(64),
                                            transforms.CenterCrop(64),
                                            #transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5),(0.5))])


    def open_dataset(self, folder, mode):
        MNIST69_DATASET = []
        mat = io.loadmat(folder)
        class_6 = []
        class_9 = []
        progress_bar = tqdm(range(100))
        for i in range(100):            
            single_recording = {'fMRIdata': mat['Y'][i],
                                'image': mat['X'][i],
                                'label': int(mat['labels'][i])}
            if int(mat['labels'][i]) == 1:
              class_6.append(i)
            elif int(mat['labels'][i]) == 2:
              class_9.append(i)

            MNIST69_DATASET.append(single_recording)
            progress_bar.update()  

        # Shuffle the original list indices
        random.shuffle(class_6)
        random.shuffle(class_9)
        # Calculate the split indices
        split_6 = int(len(class_6) * 0.9)
        split_9 = int(len(class_9) * 0.9)
        list_idx_class6_train = class_6[:split_6]
        list_idx_class6_val = class_6[split_6:]
        list_idx_class9_train = class_9[:split_9]
        list_idx_class9_val = class_9[split_9:]
        # Split the list
        data_list = []
        if mode == 'train':
          for idx in list_idx_class6_train:
              data_list.append(MNIST69_DATASET[idx])
          for idx in list_idx_class9_train:
              data_list.append(MNIST69_DATASET[idx])
        elif mode == 'val':
          for idx in list_idx_class6_val:
              data_list.append(MNIST69_DATASET[idx])
          for idx in list_idx_class9_val:
              data_list.append(MNIST69_DATASET[idx])
        elif mode == 'test':
          for idx in list_idx_class6_val:
              data_list.append(MNIST69_DATASET[idx])
          for idx in list_idx_class9_val:
              data_list.append(MNIST69_DATASET[idx])

        return data_list


    def __getitem__(self, index):

        if self.mode == 'train':
          # Take one recording from the dataset
          recording_dictionary = self.dataset[index]
          # Extract the Image [image of size (28,28)]
          image = recording_dictionary['image'].reshape(28, 28, order='F')
          #image = resize(image, (100, 100), anti_aliasing=True)
          #image = torch.tensor(image)
          image = Image.fromarray(image)
          image = self.transform(image)
          # Extract the EEG Recording [tensor of size [N,T]]
          recording_fMRI = torch.tensor(recording_dictionary['fMRIdata'])

          return image, recording_fMRI

        elif self.mode == 'val':
          # Take one recording from the dataset
          recording_dictionary = self.dataset[index]
          # Extract the Image [image of size (28,28)]
          image = recording_dictionary['image'].reshape(28, 28, order='F')
          #image = resize(image, (100, 100), anti_aliasing=True)
          #image = torch.tensor(image)
          image = Image.fromarray(image)
          image = self.transform(image)
          # Extract the EEG Recording [tensor of size [N,T]]
          recording_fMRI = torch.tensor(recording_dictionary['fMRIdata'])

          return image, recording_fMRI

        elif self.mode == 'test':
          # Take one recording from the dataset
          recording_dictionary = self.dataset[index]
          # Extract the Image [image of size (28,28)]
          image = recording_dictionary['image'].reshape(28, 28, order='F')
          #image = resize(image, (100, 100), anti_aliasing=True)
          #image = torch.tensor(image)
          image = Image.fromarray(image)
          image = self.transform(image)
          # Extract the EEG Recording [tensor of size [N,T]]
          recording_fMRI = torch.tensor(recording_dictionary['fMRIdata'])

          return image, recording_fMRI


    def __len__(self):
        return len(self.dataset)


    def vismnist_collate_fn(self, data_batch):

        data_batch.sort(key=lambda d: len(d[1]), reverse=True)
        imgs, recording_fMRI = zip(*data_batch)
        
        imgs = torch.stack(imgs, dim=0)
        recordings_fMRI = torch.stack(recording_fMRI, dim=0)

        return imgs, recordings_fMRI


class MNIST69_Data_Module(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 data_folder):
        super().__init__()
        self.batch_size = batch_size
        self.data_folder = data_folder


    def setup(self, stage=None):
        if stage in (None, 'fit'):
          self.train_ds = MNIST69_Dataset(mode='train', batch_size=self.batch_size, data_folder=self.data_folder)
          self.val_ds = MNIST69_Dataset(mode='val', batch_size=self.batch_size, data_folder=self.data_folder)
        if stage == 'predict':
          self.test_ds = MNIST69_Dataset(mode='test', batch_size=1, data_folder=self.data_folder)


    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          num_workers=2,
                          batch_size=self.batch_size,
                          collate_fn=self.train_ds.vismnist_collate_fn,
                          shuffle=True)
        

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          num_workers=2,
                          batch_size=self.batch_size,
                          collate_fn=self.val_ds.vismnist_collate_fn,
                          shuffle=False)
        

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds,
                          batch_size=self.test_ds.batch_size,
                          num_workers=2,
                          shuffle=False)



'''#############################################################'''
'''################### Architecture Modules ####################'''
'''#############################################################'''

class VisualEncoder(nn.Module):
    def __init__(self, num_channels=1, num_features=8*8*256, z_dim=128):
        super(VisualEncoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=64, kernel_size=5, stride=2, padding=2) # (50,50,64) # (32,32,64)
        self.bn1 = nn.BatchNorm2d(64, momentum=0.9) 
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=2, padding=2) # (25,25,128) # (16,16,128)
        self.bn2 = nn.BatchNorm2d(128, momentum=0.9) 
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2) # (13,13,256) # (8,8,256)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)

        self.fc1 = nn.Linear(in_features=num_features, out_features=2048)
        self.bn4 = nn.BatchNorm1d(2048,momentum=0.9)
        self.fc2 = nn.Linear(in_features=2048, out_features=z_dim)
        self.fc3 = nn.Linear(in_features=2048, out_features=z_dim)

        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        #x = x.unsqueeze(dim=1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        x = x.float()
        x = self.conv1(x) 
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x) 
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x) # (13,13,256)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)

        mu = self.fc2(x)
        logvar = self.fc3(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std


class CognitiveEncoder(nn.Module):
    def __init__(self, num_features=8428, z_dim=128):
        super(CognitiveEncoder, self).__init__()

        self.fc1 = nn.Linear(in_features=num_features, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=z_dim)
        self.fc3 = nn.Linear(in_features=1024, out_features=z_dim)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        x = x.float()
        x = self.fc1(x)
        mu = self.fc2(x)
        logvar = self.fc3(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

class Generator(nn.Module):
    def __init__(self, input_size=128):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, 8 * 8 * 256)
        self.bn1 = nn.BatchNorm1d(8 * 8 * 256, momentum=0.9)  
        self.deconv1 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=6, stride=2, padding=2) # (25,25,256)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)            
        self.deconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=6, stride=2, padding=2) # (50,50,128)
        self.bn3 = nn.BatchNorm2d(128, momentum=0.9)  
        self.deconv3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=6, stride=2, padding=2) # (100,100,32)
        self.bn4 = nn.BatchNorm2d(32, momentum=0.9)  
        self.deconv4 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2) # (100,100,1)
        self.tanh = nn.Tanh()

        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = x.float()
        x = self.fc(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = x.reshape(-1, 256, 8, 8)

        x = self.deconv1(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        x = self.deconv2(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.deconv3(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.deconv4(x)
        x = self.tanh(x)
        return x #.squeeze()

class Discriminator(nn.Module):
    def __init__(self, num_features=8*8*256):
        super(Discriminator, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2) # (100,100,32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=128, kernel_size=5, stride=2, padding=2) # (50,50,128)
        self.bn1 = nn.BatchNorm2d(128, momentum=0.9)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5, stride=2, padding=2) # (25,25,256)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.9)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=5, stride=2, padding=2) # (13,13,256)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.9)

        self.fc1 = nn.Linear(in_features=num_features, out_features=512)
        self.bn4 = nn.BatchNorm1d(512,momentum=0.9)
        self.fc2 = nn.Linear(in_features=512, out_features=1)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.float()
        #x = x.unsqueeze(dim=1)
        x = self.conv1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        x = x.reshape(x.shape[0], -1)
        fs = x
        x = self.fc1(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)

        x = self.fc2(x)
        x = self.sigmoid(x)
        
        return x, fs



'''#############################################################'''
'''################# The Proposed Architecture #################'''
'''#############################################################'''

class NeuralVisualDecodingfMRIModel(pl.LightningModule):

    def __init__(self,
                 visualencoder_hyperparameters,
                 cognitivencoder_hyperparameters,
                 generator_hyperparameters,
                 discriminator_hyperparameters):
        super(NeuralVisualDecodingfMRIModel, self).__init__()
        self.automatic_optimization = False

        # ---- Visual Encoder Hyperparameters --------------------------------------------------------------
        self.ve_num_channels = visualencoder_hyperparameters['num_channels']
        self.ve_num_features = visualencoder_hyperparameters['num_features']
        self.ve_z_dim = visualencoder_hyperparameters['z_dim']
        # ---- Cognitive Encoder Hyperparameters -----------------------------------------------------------
        self.ce_num_features = cognitivencoder_hyperparameters['num_features']
        self.ce_z_dim = cognitivencoder_hyperparameters['z_dim']
        # ---- Generator Hyperparameters -------------------------------------------------------------------
        self.g_input_size = generator_hyperparameters['input_size']
        # ---- Discriminator Hyperparameters ---------------------------------------------------------------
        self.d_num_features = discriminator_hyperparameters['num_features']

        self.visual_encoder = VisualEncoder(self.ve_num_channels, self.ve_num_features, self.ve_z_dim)
        self.cognitive_encoder = CognitiveEncoder(self.ce_num_features, self.ce_z_dim)
        self.generator = Generator(self.g_input_size)
        self.discriminator = Discriminator(self.d_num_features)

        self.visual_encoder.apply(self.weights_init)
        self.cognitive_encoder.apply(self.weights_init)
        self.generator.apply(self.weights_init)
        self.discriminator.apply(self.weights_init)

        self.criterion = nn.BCELoss()
        self.gamma = 15

    def forward(self, x):
        # Cognitive Encoder
        z_cog, _, _ = self.cognitive_encoder(x)
        #z_vis, _, _ = self.visual_encoder(x)
        # Generator
        fMRI_reconstruction = self.generator(z_cog)
        return fMRI_reconstruction

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def configure_optimizers(self):
        stage1_discriminator_optimizer = torch.optim.RMSprop(list(self.discriminator.parameters()), lr=3e-5) #, betas=(0.5, 0.999), weight_decay=0.98)
        stage1_generator_optimizer = torch.optim.RMSprop(list(self.generator.parameters()), lr=3e-4) #, betas=(0.5, 0.999), weight_decay=0.98)
        stage1_encoder_optimizer = torch.optim.RMSprop(list(self.visual_encoder.parameters()), lr=3e-4) #, betas=(0.5, 0.999), weight_decay=0.98)

        stage2_discriminator_optimizer = torch.optim.RMSprop(list(self.discriminator.parameters()), lr=3e-4) #, betas=(0.9, 0.999), weight_decay=0.98)
        stage2_encoder_optimizer = torch.optim.RMSprop(list(self.cognitive_encoder.parameters()), lr=3e-4) #, betas=(0.9, 0.999), weight_decay=0.98)

        stage3_discriminator_optimizer = torch.optim.RMSprop(list(self.discriminator.parameters()), lr=3e-4) #, betas=(0.9, 0.999), weight_decay=0.98)
        stage3_generator_optimizer = torch.optim.RMSprop(list(self.generator.parameters()), lr=3e-4) #, betas=(0.9, 0.999), weight_decay=0.98)

        return stage1_discriminator_optimizer, stage1_generator_optimizer, stage1_encoder_optimizer, stage2_discriminator_optimizer, stage2_encoder_optimizer, stage3_discriminator_optimizer, stage3_generator_optimizer

    def training_step(self, train_batch, batch_idx): 
        # Access the current epoch number
        current_epoch = self.trainer.current_epoch

        stage1_discriminator_optimizer, stage1_generator_optimizer, stage1_encoder_optimizer, stage2_discriminator_optimizer, stage2_encoder_optimizer, stage3_discriminator_optimizer, stage3_generator_optimizer = self.optimizers()     
        torch.set_grad_enabled(True)  

        imgs, recordings_fMRI = train_batch
        imgs, recordings_fMRI = imgs.to(device), recordings_fMRI.to(device)
        bs = imgs.shape[0]

        if current_epoch < 1000:
          '''######################### S T A G E 1 #########################'''
          ## ------------------- TRAIN THE DISCRIMINATOR ----------------------
          ones_label = torch.ones(bs, 1).to(device)
          zeros_label = torch.zeros(bs, 1).to(device)
          zeros_label_noise = torch.zeros(bs, 1).to(device)

          z_vis, _, _ = self.visual_encoder(imgs)
          image_reconstruction = self.generator(z_vis)
          z_noise = torch.randn(bs, 128).to(device)
          rec_noise = self.generator(z_noise)

          output = self.discriminator(imgs)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(image_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)

          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          self.log('train loss discriminator', discriminator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('train loss discriminator\t',  discriminator_loss_value)
          
          stage1_discriminator_optimizer.zero_grad()
          self.manual_backward(discriminator_loss_value, retain_graph=True)
          stage1_discriminator_optimizer.step()

          ## -------------------- TRAIN THE GENERATOR -------------------------
          output = self.discriminator(imgs)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(image_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)
          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          
          fs_tilde = self.discriminator(image_reconstruction)[1]
          fs = self.discriminator(imgs)[1]
          rec_loss = ((fs_tilde - fs)**2).mean()

          generator_loss_value = self.gamma * rec_loss - discriminator_loss_value
          self.log('train loss generator', generator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('train loss generator\t',  generator_loss_value)
          
          stage1_generator_optimizer.zero_grad()
          self.manual_backward(generator_loss_value, retain_graph=True)
          stage1_generator_optimizer.step()

          ## --------------------- TRAIN THE ENCODER --------------------------
          z_vis, mu_vis, logvar_vis = self.visual_encoder(imgs)
          image_reconstruction = self.generator(z_vis)
          
          fs_tilde = self.discriminator(image_reconstruction)[1]
          fs = self.discriminator(imgs)[1]
          rec_loss = ((fs_tilde - fs)**2).mean()
          KLD = (-0.5 * torch.sum(rec_loss))/torch.numel(mu_vis.data)

          encoder_loss_value = KLD + 5 * rec_loss
          self.log('train loss encoder', encoder_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('train loss encoder\t',  encoder_loss_value)
          
          stage1_encoder_optimizer.zero_grad()
          self.manual_backward(encoder_loss_value, retain_graph=True)
          stage1_encoder_optimizer.step()

        elif current_epoch >= 1000 and current_epoch < 2000:
          '''######################### S T A G E 2 #########################'''
          ## ------------------- TRAIN THE DISCRIMINATOR ----------------------
          ones_label = torch.ones(bs, 1).to(device)
          zeros_label = torch.zeros(bs, 1).to(device)
          zeros_label_noise = torch.zeros(bs, 1).to(device)

          z_vis, _, _ = self.visual_encoder(imgs)
          image_reconstruction = self.generator(z_vis)
          z_cog, _, _ = self.cognitive_encoder(recordings_fMRI)
          fMRI_reconstruction = self.generator(z_cog)
          z_noise = torch.randn(bs, 128).to(device)
          rec_noise = self.generator(z_noise)

          output = self.discriminator(image_reconstruction)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(fMRI_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)

          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          self.log('train loss discriminator', discriminator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('train loss discriminator\t',  discriminator_loss_value)
          
          stage2_discriminator_optimizer.zero_grad()
          self.manual_backward(discriminator_loss_value, retain_graph=True)
          stage2_discriminator_optimizer.step()

          ## --------------------- TRAIN THE ENCODER --------------------------
          z_cog, mu_cog, logvar_cog = self.cognitive_encoder(recordings_fMRI)
          fMRI_reconstruction = self.generator(z_cog)
          
          fs_tilde = self.discriminator(fMRI_reconstruction)[1]
          fs = self.discriminator(image_reconstruction)[1]
          rec_loss = ((fs_tilde - fs)**2).mean()
          KLD = (-0.5 * torch.sum(rec_loss))/torch.numel(mu_cog.data)

          encoder_loss_value = KLD + 5 * rec_loss
          self.log('train loss encoder', encoder_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('train loss encoder\t',  encoder_loss_value)
          
          stage2_encoder_optimizer.zero_grad()
          self.manual_backward(encoder_loss_value, retain_graph=True)
          stage2_encoder_optimizer.step()

        elif current_epoch >= 2000: # Till 1000 epochs
          '''######################### S T A G E 3 #########################'''
          ## ------------------- TRAIN THE DISCRIMINATOR ----------------------
          ones_label = torch.ones(bs, 1).to(device)
          zeros_label = torch.zeros(bs, 1).to(device)
          zeros_label_noise = torch.zeros(bs, 1).to(device)

          z_cog, _, _ = self.cognitive_encoder(recordings_fMRI)
          fMRI_reconstruction = self.generator(z_cog)
          z_noise = torch.randn(bs, 128).to(device)
          rec_noise = self.generator(z_noise)

          output = self.discriminator(imgs)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(fMRI_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)

          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          self.log('train loss discriminator', discriminator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('train loss discriminator\t',  discriminator_loss_value)
          
          stage3_discriminator_optimizer.zero_grad()
          self.manual_backward(discriminator_loss_value, retain_graph=True)
          stage3_discriminator_optimizer.step()

          ## -------------------- TRAIN THE GENERATOR -------------------------
          output = self.discriminator(imgs)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(fMRI_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)
          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          
          fs_tilde = self.discriminator(fMRI_reconstruction)[1]
          fs = self.discriminator(imgs)[1]
          rec_loss = ((fs_tilde - fs)**2).mean()

          generator_loss_value = self.gamma * rec_loss - discriminator_loss_value
          self.log('train loss generator', generator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('train loss generator\t',  generator_loss_value)
          
          stage3_generator_optimizer.zero_grad()
          self.manual_backward(generator_loss_value, retain_graph=True)
          stage3_generator_optimizer.step()

    def validation_step(self, val_batch, batch_idx):
        # Access the current epoch number
        current_epoch = self.trainer.current_epoch

        imgs, recordings_fMRI = val_batch
        imgs, recordings_fMRI = imgs.to(device), recordings_fMRI.to(device)
        bs = imgs.shape[0]

        if current_epoch < 1000:
          '''######################### S T A G E 1 #########################'''
          ## ------------------- VAL THE DISCRIMINATOR ----------------------
          ones_label = torch.ones(bs, 1).to(device)
          zeros_label = torch.zeros(bs, 1).to(device)
          zeros_label_noise = torch.zeros(bs, 1).to(device)

          z_vis, _, _ = self.visual_encoder(imgs)
          image_reconstruction = self.generator(z_vis)
          z_noise = torch.randn(bs, 128).to(device)
          rec_noise = self.generator(z_noise)

          output = self.discriminator(imgs)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(image_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)

          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          self.log('val loss discriminator', discriminator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('val loss discriminator\t',  discriminator_loss_value)

          ## -------------------- VAL THE GENERATOR -------------------------
          output = self.discriminator(imgs)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(image_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)
          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          
          fs_tilde = self.discriminator(image_reconstruction)[1]
          fs = self.discriminator(imgs)[1]
          rec_loss = ((fs_tilde - fs)**2).mean()

          generator_loss_value = self.gamma * rec_loss - discriminator_loss_value
          self.log('val loss generator', generator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('val loss generator\t',  generator_loss_value)

          ## --------------------- VAL THE ENCODER --------------------------
          z_vis, mu_vis, logvar_vis = self.visual_encoder(imgs)
          image_reconstruction = self.generator(z_vis)
          
          fs_tilde = self.discriminator(image_reconstruction)[1]
          fs = self.discriminator(imgs)[1]
          rec_loss = ((fs_tilde - fs)**2).mean()
          KLD = (-0.5 * torch.sum(rec_loss))/torch.numel(mu_vis.data)

          encoder_loss_value = KLD + 5 * rec_loss
          self.log('val loss encoder', encoder_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('val loss encoder\t',  encoder_loss_value)

        elif current_epoch >= 1000 and current_epoch < 2000:
          '''######################### S T A G E 2 #########################'''
          ## ------------------- VAL THE DISCRIMINATOR ----------------------
          ones_label = torch.ones(bs, 1).to(device)
          zeros_label = torch.zeros(bs, 1).to(device)
          zeros_label_noise = torch.zeros(bs, 1).to(device)

          z_vis, _, _ = self.visual_encoder(imgs)
          image_reconstruction = self.generator(z_vis)
          z_cog, _, _ = self.cognitive_encoder(recordings_fMRI)
          fMRI_reconstruction = self.generator(z_cog)
          z_noise = torch.randn(bs, 128).to(device)
          rec_noise = self.generator(z_noise)

          output = self.discriminator(image_reconstruction)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(fMRI_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)

          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          self.log('val loss discriminator', discriminator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('val loss discriminator\t',  discriminator_loss_value)

          ## --------------------- VAL THE ENCODER --------------------------
          z_cog, mu_cog, logvar_cog = self.cognitive_encoder(recordings_fMRI)
          fMRI_reconstruction = self.generator(z_cog)
          
          fs_tilde = self.discriminator(fMRI_reconstruction)[1]
          fs = self.discriminator(image_reconstruction)[1]
          rec_loss = ((fs_tilde - fs)**2).mean()
          KLD = (-0.5 * torch.sum(rec_loss))/torch.numel(mu_cog.data)

          encoder_loss_value = KLD + 5 * rec_loss
          self.log('val loss encoder', encoder_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('val loss encoder\t',  encoder_loss_value)

        elif current_epoch >= 2000: # Till 1000 epochs
          '''######################### S T A G E 3 #########################'''
          ## ------------------- VAL THE DISCRIMINATOR ----------------------
          ones_label = torch.ones(bs, 1).to(device)
          zeros_label = torch.zeros(bs, 1).to(device)
          zeros_label_noise = torch.zeros(bs, 1).to(device)

          z_cog, _, _ = self.cognitive_encoder(recordings_fMRI)
          fMRI_reconstruction = self.generator(z_cog)
          z_noise = torch.randn(bs, 128).to(device)
          rec_noise = self.generator(z_noise)

          output = self.discriminator(imgs)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(fMRI_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)

          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          self.log('val loss discriminator', discriminator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('val loss discriminator\t',  discriminator_loss_value)

          ## -------------------- VAL THE GENERATOR -------------------------
          output = self.discriminator(imgs)[0]
          errD_real = self.criterion(output, ones_label)
          output = self.discriminator(fMRI_reconstruction)[0]
          errD_rec_enc = self.criterion(output, zeros_label)
          output = self.discriminator(rec_noise)[0]
          errD_rec_noise = self.criterion(output, zeros_label_noise)
          discriminator_loss_value = errD_real + errD_rec_enc + errD_rec_noise
          
          fs_tilde = self.discriminator(fMRI_reconstruction)[1]
          fs = self.discriminator(imgs)[1]
          rec_loss = ((fs_tilde - fs)**2).mean()

          generator_loss_value = self.gamma * rec_loss - discriminator_loss_value
          self.log('val loss generator', generator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
          if batch_idx==109:
                print('val loss generator\t',  generator_loss_value)

    def predict_step(self, pred_batch, batch_idx):
        imgs, recordings_fMRI = pred_batch
        imgs, recordings_fMRI = imgs.to(device), recordings_fMRI.to(device)

        generated_img = self(recordings_fMRI)        
        return generated_img, imgs