import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as Functional

class BiaffineAttention(nn.Module): 
    """ This was born to be a Biaffine Attention layer. 
    But eventually it is not. A linear layer is enough and
    much lighter. 
    """ 
 
    def __init__(self, in_features, out_features): 
        super(BiaffineAttention, self).__init__() 
 
        self.in_features = float(in_features)
        self.out_features = float(out_features)
 
        self.linear = torch.nn.Linear(2 * in_features, out_features, bias=True) 
        
        self.init_weights()

    def init_weights(self): 
        init.normal_(self.linear.weight) 
        self.linear.weight.data.div_(self.linear.weight.data.sum()) 
 
    def forward(self, x_1, x_2): 
        attention = self.linear(torch.cat((x_1.float(), x_2.float()), dim=-1))
 
        return attention
 
    def reset_parameters(self): 
        #self.bilinear.reset_parameters() 
        self.linear.reset_parameters()

class Attention(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Attention, self).__init__()
        self.key = nn.Linear(hidden_size, output_size)
        self.value = nn.Linear(hidden_size, output_size)
        self.score = nn.Linear(output_size, 1)

    def forward(self, x):
        keys = self.key(x)
        values = self.value(x)
        scores = self.score(keys)
        attention = torch.softmax(scores, dim=-1)
        attention = attention.permute(0, 2, 1) # Recall we pass data in batch 
        context = attention @ values
        return context

