import torch
import torch.nn as nn

from src.framework.hyperparameters import *

class CNN_classificator(nn.Module):
    def __init__(self, num_classes):
        super(CNN_classificator, self).__init__()
        # Conv2D Layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=256,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=256,out_channels=256,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=4,stride=1),
            
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=5,stride=1,padding=2),
            nn.ReLU(),
            
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=5,stride=1,padding=2),
            nn.ReLU()
        )
        # Output Layer for each channel
        self.fc = nn.Linear(in_features=40*4*64, out_features=num_classes)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax()
        # Output Layer for each recording
        self.fc_final = nn.Linear(in_features=64, out_features=1)
        
    def forward(self, x):
        # x -> size [B, N, 43, 7, 3]
        x = x.permute(1,0,2,3,4)
        N = x.shape[0] # number of channels
        
        manifold_datapoints = []
        channels_distributions = []
        for n in range(N):
            # x[n] -> size [B, 43, 7, 3]
            y = x[n].permute(0,3,1,2).to(device)
            y = self.conv_layers(y.float())
            y = y.reshape(y.size(0), -1)
            manifold_datapoints.append(y)
            y = self.fc(y)
            y = self.dropout(y)
            y = self.softmax(y)
            # y -> size [B, num_classes]
            channels_distributions.append(y)

        manifold_datapoints = torch.stack(manifold_datapoints, dim=1)
        with torch.no_grad():
            manifold_datapoints = torch.mean(manifold_datapoints, dim=1)
        # channels_distributions -> size [N, B, num_classes]
        channels_distributions = torch.stack(channels_distributions, dim=0)
        # out -> size [B, num_classes, N]
        out = channels_distributions.permute(1, 2, 0)
        out = self.fc_final(out)
        out = out.squeeze()

        return out, manifold_datapoints

