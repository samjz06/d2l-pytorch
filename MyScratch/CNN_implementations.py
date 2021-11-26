import torch
import torch.nn as nn

class Flatten(nn.Module):
    def forward(self, x):
        # whats the difference of 'view' and 'reshape' again?
        return x.view(x.shape[0], -1)

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1, 1, 28, 28) # turn 28 into argument later

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        
        
        pass
    
    def forward(self, x):
        pass