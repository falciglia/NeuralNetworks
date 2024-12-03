'''#############################################################'''
'''#################### Importing libraries ####################'''
'''#############################################################'''

import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import numpy as np
import random

from src.framework.hyperparameters import *
from src.framework.dataset import *
from src.framework.RieManiSpectraNet_architecture import *
from src.utils.eval_metrics import *

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


'''#############################################################'''
'''########################## WORKFLOW #########################'''
'''#############################################################'''

################################################################################
########################## Data Module Instantiation ###########################
################################################################################

visMNIST_data_module = MindBigData2022_VisMNIST_Cap64_Data_Module(batch_size=batch_size, data_folder=data_folder)

################################################################################
############################# Model Instantiation ##############################
################################################################################

model = WeAreGoingToSeeYourMemoriesModel(bfb_hyperparameters,
                                         stream1_hyperparameters)
model.to(device)

################################################################################
############################ Trainer Instantiation #############################
################################################################################

trainer_VisMNIST = pl.Trainer(default_root_dir=training_folder_logs,
                              max_epochs=num_epochs,
                              accelerator=accelerator,
                              log_every_n_steps=n_steps_log,
                              strategy=DDPStrategy(find_unused_parameters=True),
                              devices=devices)

################################################################################
############################## CALIBRATED MODEL ################################
################################################################################

# Data Module
visMNIST_data_module.setup('fit')
print('Number of elements in train dataset: {}'.format(len(visMNIST_data_module.train_ds)))
print('Image tensor size: {}'.format(visMNIST_data_module.train_ds.__getitem__(0)[0].size()))
print('EEGrecording tensor size: {}'.format(visMNIST_data_module.train_ds.__getitem__(0)[1].size()))
print('Number of elements in train dataloader: {}'.format(len(visMNIST_data_module.train_dataloader())))
print('Number of elements in val dataset: {}'.format(len(visMNIST_data_module.val_ds)))
print('Image tensor size: {}'.format(visMNIST_data_module.val_ds.__getitem__(0)[0].size()))
print('EEGrecording tensor size: {}'.format(visMNIST_data_module.val_ds.__getitem__(0)[1].size()))
print('Number of elements in val dataloader: {}'.format(len(visMNIST_data_module.val_dataloader())))

# Training and Evaluating the MindBigData2022_VisMNIST_Cap64 Model 
# -- set ckpt_path=ckptmodel only if you already have a calibrated version of the model saved
trainer_VisMNIST.fit(model,
                     visMNIST_data_module.train_dataloader(),
                     visMNIST_data_module.val_dataloader(),
                     ckpt_path=ckptmodel)
print("\n\nFIT DONE\n\n")

################################################################################
######################### INFERENCE / PREDICTION STAGE #########################
################################################################################
# Run this section only if you have set ckpt_path=ckptmodel during the TRAINING
# STAGE. Otherwise comment all the following.

# Data Module
visMNIST_data_module.setup('predict')
print('Number of elements in test dataset: {}'.format(len(visMNIST_data_module.test_ds)))
print('Image tensor size: {}'.format(visMNIST_data_module.test_ds.__getitem__(0)[0].size()))
#print('EEGrecording tensor size: {}'.format(visMNIST_data_module.test_ds.__getitem__(0)[1].size()))
print('Number of elements in test dataloader: {}'.format(len(visMNIST_data_module.test_dataloader())))

# Predicting
predictions = trainer_VisMNIST.predict(model, 
                                       dataloaders=visMNIST_data_module.test_dataloader(),
                                       ckpt_path=ckptmodel)
'''ckpt_path (Union[str, Path, None]) – Either "best", "last", "hpc" or path to the checkpoint you wish to predict. 
   If None and the model instance was passed, use the current weights. Otherwise, the best model checkpoint from the 
   previous trainer.fit call will be loaded if a checkpoint callback is configured.'''

list_predictions = []
list_labels = []
for pred in predictions:
    list_predictions.append(pred[0].numpy())
    list_labels.append(pred[1].numpy())

################################################################################
############################## EVALUATION METRICS ##############################
################################################################################

accuracy = compute_accuracy(predictions, list_predictions, list_labels)
print("\nTHE ACCURACY on the test set IS: ", accuracy)

compute_confusion_matrix(list_predictions, list_labels, classes)