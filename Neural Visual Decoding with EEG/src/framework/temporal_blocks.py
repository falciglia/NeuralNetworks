import torch
import torch.nn as nn

import numpy as np
from scipy import signal
from scipy.stats import entropy

from src.framework.attention import *
from src.framework.hyperparameters import *

class TemporalFeatureExtraction_Module():
    def __init__(self, fs, seconds_width):
      self.hanning_windower = HanningWindower(fs, seconds_width)
      self.fature_maps_extractor = FeatureMapsExtractor()
      self.fs = fs

    def forward(self, x):
      windows = self.hanning_windower.forward(x)
      if torch.isnan(windows).any():
            print("NaN values detected in windows")
      feature_maps = self.fature_maps_extractor.forward(windows, self.fs)
      if torch.isnan(feature_maps).any():
            print("NaN values detected in feature_maps")
      return feature_maps


class TemporalFeatureProcessing_Module(nn.Module):
    def __init__(self, input_size, L, hidden_size_lstm, hidden_size_fc, dropout_rate, slope):
        """
        input_size = H*N*F
        hidden_size = 256
        """
        super(TemporalFeatureProcessing_Module, self).__init__()

        self.lstm_layer1 = nn.LSTM(input_size, hidden_size_lstm,  bidirectional=True, dtype=torch.float32)
        self.lstm_layer2 = nn.LSTM(2*hidden_size_lstm, hidden_size_lstm,  bidirectional=True, dtype=torch.float32)
        self.lstm_layer3 = nn.LSTM(2*hidden_size_lstm, hidden_size_lstm,  bidirectional=True, dtype=torch.float32)

        self.bn = nn.BatchNorm1d(hidden_size_lstm*2)
        self.relu = nn.LeakyReLU(slope)
        self.dropout = nn.Dropout(dropout_rate)

        self.attention = Attention(2*L*hidden_size_lstm, 2*L*hidden_size_lstm)

        self.fc = nn.Linear(2*2*L*hidden_size_lstm, hidden_size_fc)
        

    def forward(self, x):
        """
        input tensor x of size [H,L,N,F]
        """
        bs = x.shape[0] #batch_size (or less than batch_size)

        x = x.to(device)
        x = x.float()
        x = x.permute(0, 2, 1, 3, 4) # size [L,H,N,F]  # Recall we pass data in batch 
        x = x.reshape(x.size(0), x.size(1), -1) # size [L,input_size=H*N*F]  # Recall we pass data in batch 

        # First LSTM Layer
        x, _ = self.lstm_layer1(x) # size [L,hidden_size_lstm=256]
        x = self.bn(x.permute(0, 2, 1))
        x = self.relu(x.permute(0, 2, 1))

        # Second LSTM Layer
        x, _ = self.lstm_layer2(x) # size [L,hidden_size_lstm=256]
        x = self.bn(x.permute(0, 2, 1))
        x = self.relu(x.permute(0, 2, 1))

        # Third LSTM Layer
        x, _ = self.lstm_layer3(x) # size [L,hidden_size_lstm=256]
        x = self.bn(x.permute(0, 2, 1))
        x = self.relu(x.permute(0, 2, 1))

        x = x.reshape(bs, 1, -1) # size [1, L * hidden_size_lstm=256]
        # Attention Layer
        attention = self.attention(x) # size [1, L * hidden_size_lstm=256]
        x = torch.cat((x, attention), dim=-1) # size [1,2*L*hidden_size_lstm] 
        
        # Fully connected Layer
        x = self.fc(x) # size [1, hidden_size_fc=64]
        x = self.dropout(x)
        return x.squeeze()


class HanningWindower():
    def __init__(self, fs, seconds_width):
      super().__init__()
      self.R = int(fs*seconds_width)

    def forward(self, x):
      # Recall we pass data in batch
      B = x.shape[0] # batch size
      H = x.shape[1]
      N = x.shape[2]
      T = x.shape[3]

      batch_output = []
      for batch in range(B):

          # Stack together H [L,N,R] to reach [H,L,N,R]
          output = []
          for subband in range(H):

            # Stack together the L [N,R] to reach [L,N,R]
            windows = []
            for t in range(0, T-self.R//2, self.R//2):

              # Re-elaborate a [N,T] in L [N,R]
              segments = []
              for channel in range(N):

                segment = x[batch][subband][channel][t:t+self.R]
                segment = torch.hann_window(len(segment))*segment
                segments.append(segment)

              segments = torch.stack(segments, dim=0)
              windows.append(segments)

            windows = torch.stack(windows, dim=0)
            output.append(windows)

          output = torch.stack(output, dim=0)
          batch_output.append(output)

      batch_output = torch.stack(batch_output,dim=0)
      return batch_output


class FeatureMapsExtractor():
    def __init__(self):
      super().__init__()

    def forward(self, x, fs):
      # Recall we pass data in batch
      B = x.shape[0] # batch size
      H = x.shape[1]
      L = x.shape[2]
      N = x.shape[3]
      R = x.shape[4]

      batch_output = []
      for batch in range(B):

          output = []
          for subband in range(H):

            output_on_windows = []
            for window in range(L):

              output_on_channels = []
              for channel in range(N):

                segment = x[batch][subband][window][channel]
                # PSD feature -> It is a tensor of 101 
                f, Pxx_den = signal.welch(segment, fs=fs, nperseg=len(segment)) 
                # DE feature
                #diff = np.diff(segment)
                #de = -np.sum(np.power(diff,2)*np.log2(np.power(diff,2)))
                bins = np.histogram(segment, bins=len(segment))[1]
                hist = np.histogram(segment, bins=bins)[0]
                hist = hist / np.sum(hist)
                de = entropy(hist)

                feats_per_segment = np.append(Pxx_den, de)
                feats_per_segment = torch.tensor(feats_per_segment)
                # Replace any NaN values with zero
                feats_per_segment = torch.nan_to_num(feats_per_segment, nan=0)

                output_on_channels.append(feats_per_segment)

              output_on_channels = torch.stack(output_on_channels, dim=0)
              output_on_windows.append(output_on_channels)

            output_on_windows = torch.stack(output_on_windows, dim=0)
            output.append(output_on_windows)

          output = torch.stack(output, dim=0)
          batch_output.append(output)

      batch_output = torch.stack(batch_output,dim=0)
      return batch_output

