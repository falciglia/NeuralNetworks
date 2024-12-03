import torch
import torch.nn as nn
import torch.nn.functional as Functional

from src.framework.attention import *

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, z_dim, num_layers=2):
        super(VAE, self).__init__()
        
        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, z_dim)
        self.fc3 = nn.Linear(hidden_dim, z_dim)
        
        # Additional hidden layers (if num_layers > 2)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_layers-2):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Decoder layers
        self.fc4 = nn.Linear(z_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = Functional.relu(self.fc1(x))
        # Apply additional hidden layers, if any
        for layer in self.hidden_layers:
            h = Functional.relu(layer(h))
        return self.fc2(h), self.fc3(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decode(self, z):
        h = Functional.relu(self.fc4(z))
        # Apply additional hidden layers, if any
        for layer in self.hidden_layers:
            h = Functional.relu(layer(h))
        return torch.sigmoid(self.fc5(h))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z
    
    def loss_function(self, recon_x, x, mu, logvar):
        #BCE = Functional.binary_cross_entropy(torch.sigmoid(recon_x), x, reduction='sum')
        MSE = nn.MSELoss(reduction='mean')(torch.sigmoid(recon_x), x) # This line calculates the Kullback-Leibler divergence (KLD) between the approximate posterior distribution and the prior distribution, which is used as a regularization term in the VAE loss.
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + KLD


class FeatureFusion_Module(nn.Module):
    def __init__(self, input_size, hidden_size_fc, dropout_rate):
        """
        input_size = H*N*F
        hidden_size = 256
        """
        super(FeatureFusion_Module, self).__init__()

        self.attention = Attention(input_size, 256) #input_size//2)
        
        self.fc = nn.Linear(256*2, hidden_size_fc)
        #self.fc1 = nn.Linear(64*2, 2*hidden_size_fc)        
        #self.fc2 = nn.Linear(2*hidden_size_fc, 2*hidden_size_fc)
        #self.fc3 = nn.Linear(2*hidden_size_fc, hidden_size_fc)
        #self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        

    def forward(self, vae_s, vae_t, s, t):
        """
        vae_s # size [64]
        vae_t # size [64]
        s # size [64]
        t # size [64]
        """
        # input
        x = torch.cat((vae_s.squeeze(), vae_t.squeeze()), dim=-1) # size [input_size=2*64]    # Recall we pass data in batch

        if x.dim() != 2:
          x = x.unsqueeze(dim=0)
        if s.dim() != 2:
          s = s.unsqueeze(dim=0)
        if t.dim() != 2:
          t = t.unsqueeze(dim=0)

        # Attention Layer
        attention = self.attention(x.unsqueeze(dim=1)) # size [1,input_size]  # Recall we pass data in batch

        # Recall we pass data in batch
        B = x.shape[0] # batch size

        batch_output = []
        for batch in range(B):
            # Spatial stream
            spatial = attention[batch] @ s[batch] + s[batch]
            # Temporal stream
            temporal = attention[batch] @ t[batch] + t[batch]
            # Concatenate streams
            cnc = torch.cat((spatial.squeeze(), temporal.squeeze()), dim=0) # size [input_size, input_size] 

            batch_output.append(cnc)
        batch_output = torch.stack(batch_output,dim=0)

        # Fully connected Layers
        x = self.fc(batch_output) # size [input_size, input_size]
        #x = self.fc1(batch_output) # size [input_size, input_size]
        #x = self.relu(x)
        #x = self.dropout(x)
        
        #x = self.fc2(x)
        #x = self.relu(x)
        #x = self.dropout(x)
        
        #x = self.fc3(x)
        x = self.dropout(x)
        return x
   