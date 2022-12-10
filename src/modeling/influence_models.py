import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_influence_model(name, num_class, train_dataset_size, hidden_size):
    is_influence_model_hashmap = False
    if name == 'Net_IF':
        influence_model = Net_IF(num_class).to('cuda')
    elif name == 'MNIST_IF_1':
        if hidden_size is None:
            raise ValueError('hidden_size must be specified for MNIST_IF_1')
        influence_model = MNIST_IF_1(hidden_size, num_class).to('cuda')
    elif name == 'hashmap_IF':
        if train_dataset_size is None:
            raise ValueError('train_dataset_size must be specified for hashmap_IF')
        influence_model = hashmap_IF(train_dataset_size).to('cuda')
        is_influence_model_hashmap = True
    else:
        raise NotImplementedError()
    return influence_model, is_influence_model_hashmap


class Net_IF(nn.Module):
    def __init__(self, num_classes, out_dim=1):
        super(Net_IF, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim * num_classes)

    def forward(self, x, y):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.gather(x, 1, y.view(-1, 1))


class Linear_IF(nn.Module):
    def __init__(self, input_dim, num_class):
        super(Linear_IF, self).__init__()
        self.linear = torch.nn.Linear(input_dim, num_class)

    def forward(self, x, y):
        outputs = self.linear(x)
        return torch.gather(outputs, 1, y.unsqueeze(1))

class MNIST_IF_1(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(MNIST_IF_1, self).__init__()
        self.l1 = nn.Linear(28*28, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
  
    def forward(self, x, y):
        out = self.l1(x.reshape(-1, 28*28))
        out = self.relu(out)
        out = self.l2(out)
        return torch.gather(out, 1, y.view(-1, 1))
class hashmap_IF(nn.Module):
    def __init__(self, num_sample):
        super(hashmap_IF, self).__init__()
        self.num_sample = num_sample
        self.map = torch.nn.Parameter(torch.zeros(num_sample))

    def forward(self, id):
        return self.map[id]

    def get_all_influence(self):
        return self.map.reshape(-1)
