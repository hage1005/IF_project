import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net_IF(nn.Module):
    def __init__(self, out_dim, num_classes):
        super(Net_IF, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim*num_classes)

    def forward(self, x, y):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.gather(x, 1, y.view(-1, 1))

class GMM1D_IF(nn.Module):
     def __init__(self, input_dim, output_dim, num_class):
         super(GMM1D_IF, self).__init__()
         self.linear = torch.nn.Linear(input_dim, output_dim * num_class)
     def forward(self, x, y):
         outputs = self.linear(x)
         return torch.gather(outputs,1,y.unsqueeze(1))
