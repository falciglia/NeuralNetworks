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
from torchmetrics import StructuralSimilarityIndexMeasure

#from nn_utils import *
from nn_utils_69_NEWTRAIN import *

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
'''################# Data Module Instantiation #################'''
'''#############################################################'''

visMNIST_data_module = MNIST69_Data_Module(batch_size=batch_size, data_folder=data_folder)



'''#############################################################'''
'''############### Setting Model Hyperparameters ###############'''
'''#############################################################'''

visualencoder_hyperparameters = {'num_channels': 1,
                                 'num_features': 8*8*256,
                                 'z_dim': 128}
cognitivencoder_hyperparameters = {'num_features': 3092,
                                   'z_dim': 128}
generator_hyperparameters = {'input_size': 128}
discriminator_hyperparameters = {'num_features': 8*8*256}



'''#############################################################'''
'''#################### Model Instantiation ####################'''
'''#############################################################'''

model = NeuralVisualDecodingfMRIModel(visualencoder_hyperparameters,
                                      cognitivencoder_hyperparameters,
                                      generator_hyperparameters,
                                      discriminator_hyperparameters)
model.to(device)



'''#############################################################'''
'''################### Trainer Instantiation ###################'''
'''#############################################################'''

# Trainer Model
trainer_kay = pl.Trainer(default_root_dir='/raid/home/falcigls/NeuralNetworks/DATASET_69/Stuff_NEWTRAIN/TESTS',
                         max_epochs=2000,
                         accelerator="gpu",
                         log_every_n_steps=9)



'''#############################################################'''
'''################ T R A I N I N G ## S T A G E ###############'''
'''#############################################################'''

# Data Module
visMNIST_data_module.setup('fit')
print('\nNumber of elements in train dataset: {}'.format(len(visMNIST_data_module.train_ds)))
print('Image tensor size: {}'.format(visMNIST_data_module.train_ds.__getitem__(0)[0].size()))
print('EEGrecording tensor size: {}'.format(visMNIST_data_module.train_ds.__getitem__(0)[1].size()))
print('Number of elements in train dataloader: {}'.format(len(visMNIST_data_module.train_dataloader())))
print('\nNumber of elements in val dataset: {}'.format(len(visMNIST_data_module.val_ds)))
print('Image tensor size: {}'.format(visMNIST_data_module.val_ds.__getitem__(0)[0].size()))
print('EEGrecording tensor size: {}'.format(visMNIST_data_module.val_ds.__getitem__(0)[1].size()))
print('Number of elements in val dataloader: {}'.format(len(visMNIST_data_module.val_dataloader())))


# Training and Evaluating the NeuralVisualDecodingfMRIModel Model
trainer_kay.fit(model,
                visMNIST_data_module.train_dataloader(),
                visMNIST_data_module.val_dataloader(),
                ckpt_path='/raid/home/falcigls/NeuralNetworks/DATASET_69/Stuff_NEWTRAIN/TESTS/epoch=1999-step=90000 copy.ckpt')

print("\n\nFIT DONE\n\n")



'''#############################################################'''
'''############ P R E D I C T I O N ## S T A G E ###############'''
'''#############################################################'''

# Data Module
visMNIST_data_module.setup('predict')
print('Number of elements in test dataset: {}'.format(len(visMNIST_data_module.test_ds)))
print('Image tensor size: {}'.format(visMNIST_data_module.test_ds.__getitem__(0)[0].size()))
print('Number of elements in test dataloader: {}'.format(len(visMNIST_data_module.test_dataloader())))

# Predicting
reconstruction = trainer_kay.predict(model, dataloaders=visMNIST_data_module.test_dataloader())

# reconstruction is a list of tensors
print(reconstruction[0][0].size())

for i in range(10):
    image = reconstruction[i][0].squeeze()
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    title = '/raid/home/falcigls/NeuralNetworks/DATASET_69/Stuff_NEWTRAIN/Images/reconstruction_{}'.format(i)
    plt.savefig(title, bbox_inches='tight', pad_inches=0)

    image = reconstruction[i][1].squeeze()
    plt.imshow(image, cmap='gray')
    plt.axis("off")
    title = '/raid/home/falcigls/NeuralNetworks/DATASET_69/Stuff_NEWTRAIN/Images/real_{}'.format(i+10)
    plt.savefig(title, bbox_inches='tight', pad_inches=0)

print("\n\nPREDICT DONE\n\n")

ssim = StructuralSimilarityIndexMeasure()
for elem in reconstruction:
    pearson_corr = np.corrcoef( elem[0].numpy().reshape(4096) , elem[1].numpy().reshape(4096) )[0][1]
    ssim_corr = ssim( elem[0] , elem[1] )
    print(pearson_corr, ssim_corr)