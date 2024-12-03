import torch
import torch.nn as nn
import pytorch_lightning as pl

import seaborn as sns
import matplotlib.pyplot as plt

from src.framework.spectrogram import *
from src.framework.temporal_blocks import *
from src.framework.spatial_blocks import *
from src.framework.fusion import *
from src.framework.classifier import *

from src.framework.hyperparameters import *

from src.utils.utils import *

'''#############################################################'''
'''################# The Proposed Architecture #################'''
'''#############################################################'''

class WeAreGoingToSeeYourMemoriesModel(pl.LightningModule):

    def __init__(self,
                 bfb_hyperparameters,
                 stream1_hyperparameters):
        super(WeAreGoingToSeeYourMemoriesModel, self).__init__()
        self.automatic_optimization = False
        self.flag = 0
        # ---- Butterworth Filter Bank Hyperparameters -----------------------------------------------------
        self.order = bfb_hyperparameters['order']
        self.freq_ranges = bfb_hyperparameters['freq_ranges']
        self.f_notch = bfb_hyperparameters['f_notch']
        self.Q_notch = bfb_hyperparameters['Q_notch']
        self.fs = bfb_hyperparameters['fs']
        # ---- Stream 1 Hyperparameters --------------------------------------------------------------------
        self.fs_temporal_stream1 = stream1_hyperparameters['temporal']['fs']
        self.seconds_width_temporal_stream1 = stream1_hyperparameters['temporal']['seconds_width']
        self.input_size_temporal_stream1 = stream1_hyperparameters['temporal']['input_size']
        self.n_windows_temporal__stream1 = stream1_hyperparameters['temporal']['L']
        self.hidden_size_lstm_temporal_stream1 = stream1_hyperparameters['temporal']['hidden_size_lstm']
        self.hidden_size_fc_temporal_stream1 = stream1_hyperparameters['temporal']['hidden_size_fc']
        self.dropout_temporal_stream1 = stream1_hyperparameters['temporal']['dropout_rate']
        self.slope_temporal_stream1 = stream1_hyperparameters['temporal']['slope']
        self.R_spatial_stream1 = stream1_hyperparameters['spatial']['R']
        self.S_spatial_stream1 = stream1_hyperparameters['spatial']['S']
        self.input_size_spatial_stream1 = stream1_hyperparameters['spatial']['input_size']
        self.hidden_size_fc1_spatial_stream1 = stream1_hyperparameters['spatial']['hidden_size_fc1']
        self.hidden_size_fc2_spatial_stream1 = stream1_hyperparameters['spatial']['hidden_size_fc2']
        self.dropout_spatial_stream1 = stream1_hyperparameters['spatial']['dropout_rate']
        self.input_dim_spatialVAE_stream1 = stream1_hyperparameters['spatial_vae']['input_dim']
        self.hidden_dim_spatialVAE_stream1 = stream1_hyperparameters['spatial_vae']['hidden_dim']
        self.z_dim_spatialVAE_stream1 = stream1_hyperparameters['spatial_vae']['z_dim']
        self.num_layers_spatialVAE_stream1 = stream1_hyperparameters['spatial_vae']['num_layers']
        self.input_dim_temporalVAE_stream1 = stream1_hyperparameters['temporal_vae']['input_dim']
        self.hidden_dim_temporalVAE_stream1 = stream1_hyperparameters['temporal_vae']['hidden_dim']
        self.z_dim_temporalVAE_stream1 = stream1_hyperparameters['temporal_vae']['z_dim']
        self.num_layers_temporalVAE_stream1 = stream1_hyperparameters['temporal_vae']['num_layers']
        self.input_size_fusion_stream1 = stream1_hyperparameters['fusion']['input_size']
        self.hidden_size_fc_fusion_stream1 = stream1_hyperparameters['fusion']['hidden_size_fc']
        self.dropout_fusion_stream1 = stream1_hyperparameters['fusion']['dropout_rate']
                
        self.saved_tensors_training = []
        self.classes_de_epoch = []

        # Pre-processing Butterworth Filter Bank
        self.bfb = ButterworthFilterBank(order=self.order, cut_frequencies=self.freq_ranges, f_notch=self.f_notch, Q_notch=self.Q_notch, fs=self.fs)
        # Spectrogram Extractor
        self.spectrogram = Spectrogram_Module(n_fft=256, hop_length=64)
        # Stream 1 - All EEG Recordings
        self.tf_extractor_stream1 = TemporalFeatureExtraction_Module(fs=self.fs_temporal_stream1, seconds_width=self.seconds_width_temporal_stream1)
        self.tf_processor_stream1 = TemporalFeatureProcessing_Module(input_size=self.input_size_temporal_stream1,
                                                                     L=self.n_windows_temporal__stream1,
                                                                     hidden_size_lstm=self.hidden_size_lstm_temporal_stream1,
                                                                     hidden_size_fc=self.hidden_size_fc_temporal_stream1,
                                                                     dropout_rate=self.dropout_temporal_stream1,
                                                                     slope=self.slope_temporal_stream1)
        self.sf_extractor_stream1 = SpatialFeatureExtraction_Module(R=self.R_spatial_stream1, S=self.S_spatial_stream1)
        self.sf_processor_stream1 = SpatialFeatureProcessing_Module(input_size=self.input_size_spatial_stream1,
                                                                    hidden_size_fc1=self.hidden_size_fc1_spatial_stream1,
                                                                    hidden_size_fc2=self.hidden_size_fc2_spatial_stream1,
                                                                    dropout_rate=self.dropout_spatial_stream1)
        self.vae_spatial_stream1 = VAE(input_dim=self.input_dim_spatialVAE_stream1,
                                       hidden_dim=self.hidden_dim_spatialVAE_stream1, 
                                       z_dim=self.z_dim_spatialVAE_stream1, 
                                       num_layers=self.num_layers_spatialVAE_stream1)
        self.vae_temporal_stream1 = VAE(input_dim=self.input_dim_temporalVAE_stream1, 
                                        hidden_dim=self.hidden_dim_temporalVAE_stream1, 
                                        z_dim=self.z_dim_temporalVAE_stream1, 
                                        num_layers=self.num_layers_temporalVAE_stream1)
        self.ff_stream1 = FeatureFusion_Module(input_size=self.input_size_fusion_stream1,
                                               hidden_size_fc=self.hidden_size_fc_fusion_stream1,
                                               dropout_rate=self.dropout_fusion_stream1)  
        self.biaffine_scorer = BiaffineAttention(in_features=43*7*3, 
                                                 out_features=43*7*3)    
        # Classificator
        self.classificator = CNN_classificator(num_classes=11)

        self.criterion = nn.CrossEntropyLoss()


    def forward(self, x):
        sig = x[0]
        occipital_sig = x[1]
        bs = sig.shape[0]

        # Preprocessing sig
        y, y_spec = self.bfb(sig)
        # Spectrogram
        spec_y = self.spectrogram(y_spec)
        # Stream 1
        tf_stream1 = self.tf_extractor_stream1.forward(y)
        tf_stream1 = self.tf_processor_stream1.forward(tf_stream1)
        sf_stream1 = self.sf_extractor_stream1.forward(y)
        sf_stream1 = self.sf_processor_stream1.forward(sf_stream1)
        #vae_spatial_stream1, _, _, ls_vae_spatial_stream1 = self.vae_spatial_stream1.forward(sf_stream1)
        #vae_temporal_stream1, _, _, ls_vae_temporal_stream1 = self.vae_temporal_stream1.forward(tf_stream1)
        _, _, _, ls_vae_spatial_stream1 = self.vae_spatial_stream1.forward(sf_stream1)
        _, _, _, ls_vae_temporal_stream1 = self.vae_temporal_stream1.forward(tf_stream1)
        ffeatures_stream1 = self.ff_stream1.forward(ls_vae_spatial_stream1, ls_vae_temporal_stream1, sf_stream1, tf_stream1)
        self.signal = ffeatures_stream1
        self.signal = self.signal.reshape(self.signal.shape[0], 43, 7, 3)
        self.signal = self.signal.unsqueeze(dim=1)
        x1 = spec_y.reshape(spec_y.shape[0], spec_y.shape[1], 43*7*3).to(device) 
        x2 = self.signal.reshape(self.signal.shape[0], 1, 43*7*3).to(device) 
        x2 = x2.expand(self.signal.shape[0], 64, 43*7*3).to(device)
        self.signal = self.biaffine_scorer(x1, x2) 
        #print(self.signal.size()) 
        self.signal = self.signal.reshape(self.signal.shape[0], 64, 43, 7, 3) ## 65 -> 64 + 1

        # Classificator
        prediction, _ = self.classificator(self.signal)

        return prediction


    def configure_optimizers(self):
        temporal_optimizer_stream1 = torch.optim.Adam(list(self.tf_processor_stream1.parameters()) +
                                                      list(self.vae_temporal_stream1.parameters()), lr=1e-5)
        spatial_optimizer_stream1 = torch.optim.Adam(list(self.sf_processor_stream1.parameters()) +
                                                     list(self.vae_spatial_stream1.parameters()), lr=1e-5)
        classificator_optimizer = torch.optim.SGD(list(self.ff_stream1.parameters()) +
                                                  list(self.biaffine_scorer.parameters())+
                                                   list(self.classificator.parameters()),  lr=1e-2, weight_decay=1e-6)
        return temporal_optimizer_stream1, spatial_optimizer_stream1, classificator_optimizer
    
    def training_step(self, train_batch, batch_idx):
        if self.current_epoch < 106: 
           self.training_step_1(train_batch, batch_idx)
        else:
           self.training_step_2(train_batch, batch_idx)

    def training_step_1(self, train_batch, batch_idx): 
        temporal_optimizer_stream1, spatial_optimizer_stream1, classificator_optimizer = self.optimizers()
        torch.set_grad_enabled(True)  

        imgs, recordings_EEG, classes_de_batch, labels = train_batch
        imgs, recordings_EEG[0], recordings_EEG[1], classes_de_batch, labels = imgs.to(device), recordings_EEG[0].to(device), recordings_EEG[1].to(device), classes_de_batch.to(device), labels.to(device)
        self.classes_de_epoch.append(classes_de_batch)
        bs = imgs.shape[0]


        # Preprocessing sig
        total_y, total_y_spec = self.bfb(recordings_EEG[0])
        occipital_y, occipital_y_spec = self.bfb(recordings_EEG[1])
        # Spectrogram
        spec_y = self.spectrogram(total_y_spec)

        # TRAIN TEMPORAL LAYER
        tf_stream1 = self.tf_extractor_stream1.forward(total_y)
        tf_stream1 = self.tf_processor_stream1.forward(tf_stream1.clone())
        vae_temporal_stream1, mu_temporal_stream1, logvar_temporal_stream1, ls_vae_temporal_stream1 = self.vae_temporal_stream1.forward(tf_stream1.clone())

        self.vae_temporal_stream1_loss = self.vae_temporal_stream1.loss_function(vae_temporal_stream1.clone(), tf_stream1.clone(), mu_temporal_stream1.clone(), logvar_temporal_stream1.clone())
        self.log('vae_temporal_stream1_loss_train_loss', self.vae_temporal_stream1_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)          
        
        temporal_optimizer_stream1.zero_grad()
        self.manual_backward(self.vae_temporal_stream1_loss, retain_graph=True)
        

        # TRAIN SPATIAL LAYER
        sf_stream1 = self.sf_extractor_stream1.forward(total_y)
        sf_stream1 = self.sf_processor_stream1.forward(sf_stream1.clone())
        vae_spatial_stream1, mu_spatial_stream1, logvar_spatial_stream1, ls_vae_spatial_stream1 = self.vae_spatial_stream1.forward(sf_stream1.clone())

        self.vae_spatial_stream1_loss = self.vae_spatial_stream1.loss_function(vae_spatial_stream1.clone(), sf_stream1.clone(), mu_spatial_stream1.clone(), logvar_spatial_stream1.clone())
        self.log('vae_spatial_stream1_loss_train_loss', self.vae_spatial_stream1_loss, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)

        spatial_optimizer_stream1.zero_grad()
        self.manual_backward(self.vae_spatial_stream1_loss, retain_graph=True)
        

        # TRAIN CLASSIFICATOR
        ffeatures_stream1 = self.ff_stream1.forward(ls_vae_spatial_stream1, ls_vae_temporal_stream1, sf_stream1, tf_stream1)
        self.signal = ffeatures_stream1 
        self.signal = self.signal.reshape(self.signal.shape[0], 43, 7, 3)
        self.signal = self.signal.unsqueeze(dim=1)
        #self.signal = torch.cat((spec_y.to(device), self.signal.to(device)), dim=1).to(device)
        x1 = spec_y.reshape(spec_y.shape[0], spec_y.shape[1], 43*7*3).to(device) 
        x2 = self.signal.reshape(self.signal.shape[0], 1, 43*7*3).to(device) 
        x2 = x2.expand(self.signal.shape[0], 64, 43*7*3).to(device)
        self.signal = self.biaffine_scorer(x1, x2) 
        #print(self.signal.size()) # To see if it is equal to [B, 64, 903] or [B, 1, 903] or what else 
        self.signal = self.signal.reshape(self.signal.shape[0], 64, 43, 7, 3) ## 65 -> 64 + 1


        predictions, manifold_datapoints = self.classificator(self.signal)
        # Manifold
        self.saved_tensors_training.append(manifold_datapoints) 

        # Graphs and Metrics
        preds = torch.argmax(predictions, dim=-1) 
        correct = torch.sum(preds == labels.data)
        accuracy = correct.double() / batch_size

        classificator_loss_value = self.criterion(predictions, labels)
        self.log('train loss', classificator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('train acc', accuracy, on_epoch=True, prog_bar=True, logger=True)
       
        classificator_optimizer.zero_grad()
        self.manual_backward(classificator_loss_value, retain_graph=True)
        
        # Switch training phases
        if abs(self.vae_spatial_stream1_loss) < 0.15 and abs(self.vae_temporal_stream1_loss) < 0.15:
           if batch_idx == 21:
            print("STAI FUORI")
            self.flag+=1

        temporal_optimizer_stream1.step() 
        spatial_optimizer_stream1.step()
        classificator_optimizer.step()

        # Empty GPU cache
        torch.cuda.empty_cache()
        
    def training_step_2(self, train_batch, batch_idx): 
        _, _, classificator_optimizer_total = self.optimizers()
        torch.set_grad_enabled(True)  

        imgs, recordings_EEG, classes_de_batch, labels = train_batch
        imgs, recordings_EEG[0], recordings_EEG[1], classes_de_batch, labels = imgs.to(device), recordings_EEG[0].to(device), recordings_EEG[1].to(device), classes_de_batch.to(device), labels.to(device)
        self.classes_de_epoch.append(classes_de_batch)
        bs = imgs.shape[0]


        # Preprocessing sig
        total_y, total_y_spec = self.bfb(recordings_EEG[0])
        occipital_y, occipital_y_spec = self.bfb(recordings_EEG[1])
        # Spectrogram
        spec_y = self.spectrogram(total_y_spec)

        # TEMPORAL LAYER
        tf_stream1 = self.tf_extractor_stream1.forward(total_y)
        tf_stream1 = self.tf_processor_stream1.forward(tf_stream1.clone())
        vae_temporal_stream1, mu_temporal_stream1, logvar_temporal_stream1, ls_vae_temporal_stream1 = self.vae_temporal_stream1.forward(tf_stream1.clone())

        # SPATIAL LAYER
        sf_stream1 = self.sf_extractor_stream1.forward(total_y)
        sf_stream1 = self.sf_processor_stream1.forward(sf_stream1.clone())
        vae_spatial_stream1, mu_spatial_stream1, logvar_spatial_stream1, ls_vae_spatial_stream1 = self.vae_spatial_stream1.forward(sf_stream1.clone())

        # TRAIN CLASSIFICATOR
        ffeatures_stream1 = self.ff_stream1.forward(ls_vae_spatial_stream1, ls_vae_temporal_stream1, sf_stream1, tf_stream1)
        self.signal = ffeatures_stream1 
        self.signal = self.signal.reshape(self.signal.shape[0], 43, 7, 3)
        self.signal = self.signal.unsqueeze(dim=1)
        #self.signal = torch.cat((spec_y.to(device), self.signal.to(device)), dim=1).to(device)
        x1 = spec_y.reshape(spec_y.shape[0], spec_y.shape[1], 43*7*3).to(device) 
        x2 = self.signal.reshape(self.signal.shape[0], 1, 43*7*3).to(device) 
        x2 = x2.expand(self.signal.shape[0], 64, 43*7*3).to(device)
        self.signal = self.biaffine_scorer(x1, x2) 
        #print(self.signal.size()) # To see if it is equal to [B, 64, 903] or [B, 1, 903] or what else 
        self.signal = self.signal.reshape(self.signal.shape[0], 64, 43, 7, 3) ## 65 -> 64 + 1

        predictions, manifold_datapoints = self.classificator(self.signal)
        # Manifold
        self.saved_tensors_training.append(manifold_datapoints)

        # Graphs and Metrics
        preds = torch.argmax(predictions, dim=-1) 
        correct = torch.sum(preds == labels.data)
        accuracy = correct.double() / batch_size

        classificator_loss_value = self.criterion(predictions, labels)
        self.log('train loss', classificator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('train acc', accuracy, on_epoch=True, prog_bar=True, logger=True)    

        classificator_optimizer_total.zero_grad()
        self.manual_backward(classificator_loss_value, retain_graph=True)
        
        classificator_optimizer_total.step()        

        # Empty GPU cache
        torch.cuda.empty_cache()

    def validation_step(self, val_batch, batch_idx):
        torch.set_grad_enabled(False)
        
        imgs, recordings_EEG, classes_de_batch, labels = val_batch
        imgs, recordings_EEG[0], recordings_EEG[1], classes_de_batch, labels = imgs.to(device), recordings_EEG[0].to(device), recordings_EEG[1].to(device), classes_de_batch.to(device), labels.to(device)
        bs = imgs.shape[0]


        predictions = self(recordings_EEG)
        
        # Graphs and Metrics
        preds = torch.argmax(predictions, dim=-1) 
        correct = torch.sum(preds == labels.data)
        val_accuracy = correct.double() / batch_size

        classificator_loss_value = self.criterion(predictions, labels)
        self.log('val loss', classificator_loss_value, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log('val acc', val_accuracy, on_epoch=True, prog_bar=True, logger=True)       
        return classificator_loss_value
    
    def predict_step(self, pred_batch, batch_idx):
        imgs, recordings_EEG, classes_de_batch, labels = pred_batch
        imgs, recordings_EEG[0], recordings_EEG[1], classes_de_batch, labels = imgs.to(device), recordings_EEG[0].to(device), recordings_EEG[1].to(device), classes_de_batch.to(device), labels.to(device)

        generated_img = self(recordings_EEG)        
        return generated_img, labels
    
    def on_train_epoch_end(self):
        
        concatenated_tensor = torch.cat(self.saved_tensors_training, dim=0)
        #print(concatenated_tensor)
        cnc_classes_epoch = torch.cat(self.classes_de_epoch)
        #print("Latent Space without noise size: ", concatenated_tensor.size())
        self.plot_neural_manifold(concatenated_tensor[:,0:3].detach().cpu().numpy(), '/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/RieManiSpectraNet_going_to_github/saved_pics/neural_manifold_0_ALL_ablation_2.png', cnc_classes_epoch.cpu())
        self.plot_neural_manifold(concatenated_tensor[:,746:749].detach().cpu().numpy(), '/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/RieManiSpectraNet_going_to_github/saved_pics/neural_manifold_1_ALL_ablation_2.png', cnc_classes_epoch.cpu())
        self.plot_neural_manifold(concatenated_tensor[:,114:117].detach().cpu().numpy(), '/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/RieManiSpectraNet_going_to_github/saved_pics/neural_manifold_2_ALL_ablation_2.png', cnc_classes_epoch.cpu())
        self.plot_neural_manifold(concatenated_tensor[:,819:822].detach().cpu().numpy(), '/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/RieManiSpectraNet_going_to_github/saved_pics/neural_manifold_3_ALL_ablation_2.png', cnc_classes_epoch.cpu())
        self.plot_neural_manifold(concatenated_tensor[:,442:445].detach().cpu().numpy(), '/home/s.falciglia/copia-da-prod/salvatore-backup-advanced/RieManiSpectraNet_going_to_github/saved_pics/neural_manifold_4_ALL_ablation_2.png', cnc_classes_epoch.cpu())

        # reset the list for the next epoch
        self.saved_tensors_training = []
        self.classes_de_epoch = []
        
    def plot_neural_manifold(self, x_embd, title, classes):
        if np.max(np.abs(x_embd)) != 0:
            x_embd = x_embd / np.max(np.abs(x_embd)) # normalise the values
        AXIS_LIM = np.max(x_embd)

        ds_plt = 1 # downsample for plotting
        down = 3 # downsampling param for manifold embedding

        ### Plot embedding
        plt.set_cmap('hsv') # circular cmap
        MIN = -AXIS_LIM
        fig = plt.figure(figsize=(12,12))
        grid = fig.add_gridspec(ncols=3, nrows=3)
        plt.suptitle('Neural Manifold')

        # 3D projection
        ax = fig.add_subplot(grid[1:,:], projection='3d')
        cmap = classes
        scat = ax.scatter(x_embd[:,0][::ds_plt], x_embd[:,1][::ds_plt], x_embd[:,2][::ds_plt], alpha=.7, c=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Comp1 vs Comp2
        ax = fig.add_subplot(grid[0,0])
        plt.scatter(x_embd[:,0][::ds_plt], x_embd[:,1][::ds_plt], alpha=.7, c=cmap)
        ax.set_xlabel('Comp 1'); plt.ylabel('Comp 2')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim([-AXIS_LIM-.1,AXIS_LIM+.1]); ax.set_ylim([-AXIS_LIM-.1,AXIS_LIM+.1])
        sns.despine()
        # Comp2 vs Comp3
        ax = fig.add_subplot(grid[0,1])
        plt.scatter(x_embd[:,1][::ds_plt], x_embd[:,2][::ds_plt], alpha=.7, c=cmap)
        ax.set_xlabel('Comp 2'); plt.ylabel('Comp 3')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim([-AXIS_LIM-.1,AXIS_LIM+.1]); ax.set_ylim([-AXIS_LIM-.1,AXIS_LIM+.1])
        sns.despine()
        # Comp1 vs Comp3
        ax = fig.add_subplot(grid[0,2])
        plt.scatter(x_embd[:,0][::ds_plt], x_embd[:,2][::ds_plt], alpha=.7, c=cmap)
        ax.set_xlabel('Comp 1'); plt.ylabel('Comp 3')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim([-AXIS_LIM-.1,AXIS_LIM+.1]); ax.set_ylim([-AXIS_LIM-.1,AXIS_LIM+.1])
        sns.despine()

        fig.savefig(title)

        plt.close()
        