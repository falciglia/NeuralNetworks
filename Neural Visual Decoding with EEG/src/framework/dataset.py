import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import numpy as np
from tqdm.notebook import tqdm
import random
from skimage.transform import resize

class MindBigData2022_VisMNIST_Cap64_Dataset(Dataset):

    def __init__(self,
                 batch_size,
                 mode,
                 data_folder):
        self.batch_size = batch_size
        self.mode = mode
        self.data_folder = data_folder

        self.dataset = self.open_dataset(self.data_folder, self.mode)


    def open_dataset(self, folder, mode):
        MindBigData2022_VisMNIST_Cap64_DATASET = []
        cnt_black_images = 0
        with open(folder) as file:
            lines = file.readlines()
            progress_bar = tqdm(range(len(lines)))
            for line in lines:
                list_of_recordings_2sec_200Hz = line.split(',')
                single_recording = {'dataset': list_of_recordings_2sec_200Hz[0],
                                    'origin': int(list_of_recordings_2sec_200Hz[1]),
                                    'digit_event': int(list_of_recordings_2sec_200Hz[2]),
                                    'original_png': np.array(list(map(int, list_of_recordings_2sec_200Hz[3:787]))),
                                    'timestamp': list_of_recordings_2sec_200Hz[787],
                                    'EEGdata':{}
                                    }
                k = 0           # FP1, FP2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, O2
                for channel in {'FP1', 'FPz', 'FP2', 'AF3', 'AFz', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5',
                                'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'CCPz', 'C2', 'C4', 'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1',
                                'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
                                'CB1', 'O1', 'Oz', 'O2', 'CB2'}:
                    start = 788 + 400*k
                    end = start + 400
                    single_recording['EEGdata'][channel] = np.array(list(map(float, list_of_recordings_2sec_200Hz[start:end])))
                    k = k + 1

                if single_recording['digit_event'] != -1:
                  MindBigData2022_VisMNIST_Cap64_DATASET.append(single_recording)
                else:
                  if cnt_black_images < 50:
                    MindBigData2022_VisMNIST_Cap64_DATASET.append(single_recording)
                    cnt_black_images = cnt_black_images + 1

                progress_bar.update()

        # Shuffle the original list
        random.shuffle(MindBigData2022_VisMNIST_Cap64_DATASET)
        # Calculate the split indices
        split_1 = int(len(MindBigData2022_VisMNIST_Cap64_DATASET) * 0.85)
        split_2 = int(len(MindBigData2022_VisMNIST_Cap64_DATASET) * 1)
        # Split the list
        if mode == 'train':
          data_list = MindBigData2022_VisMNIST_Cap64_DATASET[:split_1]
        elif mode == 'val':
          data_list = MindBigData2022_VisMNIST_Cap64_DATASET[split_1:split_2]
        elif mode == 'test':
          data_list = MindBigData2022_VisMNIST_Cap64_DATASET[split_1:split_2]

        return data_list


    def __getitem__(self, index):

        if self.mode == 'train':
          # Take one recording from the dataset
          recording_dictionary = self.dataset[index]
          # Extract the Image [image of size (28,28)]
          image = recording_dictionary['original_png'].reshape(28, 28)
          image = resize(image, (64, 64), anti_aliasing=True)
          image = torch.tensor(image)
          # Extract the EEG Recording [tensor of size [N,T]]
          recording_EEG = []
          occipital_recording_EEG = []
          for key in list(recording_dictionary['EEGdata'].keys()):
              recording_EEG.append(torch.tensor(recording_dictionary['EEGdata'][key]))
              if key in {'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'}:
                occipital_recording_EEG.append(torch.tensor(recording_dictionary['EEGdata'][key]))
          recording_EEG = torch.stack(recording_EEG, dim=0)
          occipital_recording_EEG = torch.stack(occipital_recording_EEG, dim=0)
          class_digitevent = torch.tensor(recording_dictionary['digit_event'])
          if recording_dictionary['digit_event'] == -1:
            label_digit_event = 10
          else:
            label_digit_event = recording_dictionary['digit_event']
          label_digit_event = torch.tensor(label_digit_event)

          return image, recording_EEG, occipital_recording_EEG, class_digitevent, label_digit_event

        elif self.mode == 'val':
          # Take one recording from the dataset
          recording_dictionary = self.dataset[index]
          # Extract the Image [image of size (28,28)]
          image = recording_dictionary['original_png'].reshape(28, 28)
          image = resize(image, (64, 64), anti_aliasing=True)
          image = torch.tensor(image)
          # Extract the EEG Recording [tensor of size [N,T]]
          recording_EEG = []
          occipital_recording_EEG = []
          for key in list(recording_dictionary['EEGdata'].keys()):
              recording_EEG.append(torch.tensor(recording_dictionary['EEGdata'][key]))
              if key in {'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'}:
                occipital_recording_EEG.append(torch.tensor(recording_dictionary['EEGdata'][key]))
          recording_EEG = torch.stack(recording_EEG, dim=0)
          occipital_recording_EEG = torch.stack(occipital_recording_EEG, dim=0)
          class_digitevent = torch.tensor(recording_dictionary['digit_event'])
          if recording_dictionary['digit_event'] == -1:
            label_digit_event = 10
          else:
            label_digit_event = recording_dictionary['digit_event']
          label_digit_event = torch.tensor(label_digit_event)

          return image, recording_EEG, occipital_recording_EEG, class_digitevent, label_digit_event

        elif self.mode == 'test':
          # Take one recording from the dataset
          recording_dictionary = self.dataset[index]
          # Extract the Image [image of size (28,28)]
          image = recording_dictionary['original_png'].reshape(28, 28)
          image = resize(image, (64, 64), anti_aliasing=True)
          image = torch.tensor(image)
          # Extract the EEG Recording [tensor of size [N,T]]
          recording_EEG = []
          occipital_recording_EEG = []
          for key in list(recording_dictionary['EEGdata'].keys()):
              recording_EEG.append(torch.tensor(recording_dictionary['EEGdata'][key]))
              if key in {'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'Oz', 'O2', 'CB2'}:
                occipital_recording_EEG.append(torch.tensor(recording_dictionary['EEGdata'][key]))
          recording_EEG = torch.stack(recording_EEG, dim=0)
          occipital_recording_EEG = torch.stack(occipital_recording_EEG, dim=0)
          class_digitevent = torch.tensor(recording_dictionary['digit_event'])
          if recording_dictionary['digit_event'] == -1:
            label_digit_event = 10
          else:
            label_digit_event = recording_dictionary['digit_event']
          label_digit_event = torch.tensor(label_digit_event)

          return image, [recording_EEG, occipital_recording_EEG], class_digitevent, label_digit_event


    def __len__(self):
        return len(self.dataset)


    def vismnist_collate_fn(self, data_batch):

        data_batch.sort(key=lambda d: len(d[1]), reverse=True)
        imgs, recording_EEG, occipital_recording_EEG, class_digitevent, labels = zip(*data_batch)
        
        imgs = torch.stack(imgs, dim=0)
        recordings_EEG = torch.stack(recording_EEG, dim=0)
        occipital_recordings_EEG = torch.stack(occipital_recording_EEG, dim=0)
        classes_de = torch.stack(class_digitevent, dim=0)
        labels = torch.stack(labels, dim=0)

        return imgs, [recordings_EEG, occipital_recordings_EEG], classes_de, labels      


class MindBigData2022_VisMNIST_Cap64_Data_Module(pl.LightningDataModule):

    def __init__(self,
                 batch_size,
                 data_folder="/home/salvatore.falciglia/pr_thesis/DATASET/MindBigData64_Mnist2022-EEGv0.016.txt"):
        super().__init__()
        self.batch_size = batch_size
        self.data_folder = data_folder


    def setup(self, stage=None):
        if stage in (None, 'fit'):
          self.train_ds = MindBigData2022_VisMNIST_Cap64_Dataset(mode='train', batch_size=self.batch_size, data_folder=self.data_folder)
          self.val_ds = MindBigData2022_VisMNIST_Cap64_Dataset(mode='val', batch_size=self.batch_size, data_folder=self.data_folder)
        if stage == 'predict':
          self.test_ds = MindBigData2022_VisMNIST_Cap64_Dataset(mode='test', batch_size=1, data_folder=self.data_folder)


    def train_dataloader(self):
        return DataLoader(dataset=self.train_ds,
                          num_workers=16,
                          batch_size=self.batch_size,
                          collate_fn=self.train_ds.vismnist_collate_fn,
                          shuffle=True)
        

    def val_dataloader(self):
        return DataLoader(dataset=self.val_ds,
                          num_workers=16,
                          batch_size=self.batch_size,
                          collate_fn=self.val_ds.vismnist_collate_fn,
                          shuffle=False)
        

    def test_dataloader(self):
        return DataLoader(dataset=self.test_ds,
                          batch_size=self.test_ds.batch_size,
                          num_workers=16,
                          shuffle=False)

