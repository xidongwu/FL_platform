# imports
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

class DNNModel(nn.Module):
 
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
#         # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        
#         # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)

#         self.linear1 = nn.Linear(in_size, hidden_size)
#         self.linear1.weight = torch.nn.Parameter(torch.ones(self.linear1.weight.size()) * 0.02)
#         self.linear1.bias = torch.nn.Parameter(torch.ones(self.linear1.bias.size()) * 0.02)
#         # output layer
#         self.linear2 = nn.Linear(hidden_size, out_size)
#         self.linear2.weight = torch.nn.Parameter(torch.ones(self.linear2.weight.size()) * 0.02)
#         self.linear2.bias = torch.nn.Parameter(torch.ones(self.linear2.bias.size()) * 0.02)
   
    def forward(self, xb):

        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out
    
    def training_step(self, images, labels):
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
