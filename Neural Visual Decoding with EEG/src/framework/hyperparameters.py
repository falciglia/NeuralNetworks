import torch

SEED = 2206

batch_size = 16
device = torch.device("cuda") #'cuda'
data_folder = "/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/RieManiSpectraNet_going_to_github/data/DATASET/MindBigData64_Mnist2022-EEGv0.016.txt"

bfb_hyperparameters = {'order': 5,
                       'freq_ranges': [14, 71], #[1, 20, 40, 60, 80, 90],
                       'f_notch': 50,
                       'Q_notch': 0.01,
                       'fs': 200}
stream1_hyperparameters = {'temporal': {'fs': 200,
                                        'seconds_width': 1,
                                        'input_size': 1*64*102, # H*N*F
                                        'L': 3,
                                        'hidden_size_lstm': 256,
                                        'hidden_size_fc': 256, #64,
                                        'dropout_rate': 0.5,
                                        'slope': 0.3},
                           'spatial': {'R': 100,
                                       'S': 8,
                                       'input_size': 1*4*36, #H*P*Supper
                                       'hidden_size_fc1': 512,
                                       'hidden_size_fc2': 256, #64,
                                       'dropout_rate': 0.5},
                           'spatial_vae': {'input_dim': 256, #64,
                                           'hidden_dim': 512,
                                           'z_dim': 64,#16,
                                           'num_layers': 3},
                           'temporal_vae': {'input_dim': 256, #64,
                                            'hidden_dim': 512,
                                            'z_dim': 64,#16,
                                            'num_layers': 3},
                           'fusion': {'input_size': 128, #32, #2*output_vae_spatial.shape[0]
                                      'hidden_size_fc': 129*7, #128,
                                      'dropout_rate': 0.5}}

training_folder_logs = "/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/RieManiSpectraNet_going_to_github/data/training_logs"
num_epochs = 1000 #1000
accelerator = 'gpu'
n_steps_log = 9
devices = 1

ckptmodel = "/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/RieManiSpectraNet_going_to_github/data/calibratedmodel/RieManiSpectraNet_calibrated.ckpt"

classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'black']