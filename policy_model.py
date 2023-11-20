import torch
from torch import nn
import torch.nn.functional as F

class GradientPolicy(nn.Module):
  def __init__(self, in_features, out_dims, hidden_size = 128):
    super().__init__()

    self.fc1 = nn.Linear(in_features, hidden_size)
    self.fc2 = nn.Linear(hidden_size, hidden_size)
    self.fc_mu = nn.Linear(hidden_size, out_dims)
    self.fc_std = nn.Linear(hidden_size, out_dims)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    loc = self.fc_mu(x)
    loc = torch.tanh(loc)#mean action distribution is between the tanh [-1,1]
    scale = self.fc_std(x) 
    scale = F.softplus(scale) + 0.001 #standrad distribution which allows positive values
    return loc, scale