from typing import ForwardRef
import torch
import torch.nn as nn
import torch.optim as optim

class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()
        self.P = nn.Embedding() # user-to-concept
        self.Q = nn.Embedding() # movie-to-concept
        self.b_u = nn.Embedding() # user bias
        self.b_i = nn.Embedding() # movie bias
        
    def forward(self, user_id, movie_id):
        P_u = self.P(user_id)
        Q_i = self.Q(movie_id)
        b_u = self.b_u(user_id)
        b_i = self.b_i(movie_id)
        result = 0
        return result
        