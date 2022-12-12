import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models, transforms


# 1 layer cnn for mnist
def get_classification_model(name, num_class):
    if name == 'Resnet34':
        classification_model = models.resnet34(pretrained=True).to('cuda')
        classification_model.fc = torch.nn.Linear(
            classification_model.fc.in_features,
            num_class).to('cuda')
    elif name == 'CnnCifar':
        classification_model = CnnCifar(num_class).to('cuda')
    elif name == 'MNIST_1':
        classification_model = MNIST_1(100, num_class).to('cuda')
    elif name == 'MNIST_2':
        classification_model = MNIST_2(num_class).to('cuda')
    elif name == 'CnnMnist':
        classification_model = CnnMnist(num_class).to('cuda')
    elif name == 'Logistic_Regression_2D':
        classification_model = Logistic_Regression_2D(num_class).to('cuda')
    else:
        raise NotImplementedError()
    return classification_model

class CnnMnist(nn.Module):
    def __init__(self, out_dim):
        super(CnnMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5)
        self.fc1 = nn.Linear(160, 50)
        self.fc2 = nn.Linear(50, out_dim)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 160)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class CnnCifar(nn.Module):
    def __init__(self, out_dim):
        super(CnnCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, out_dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Logistic_Regression_2D(nn.Module):
    def __init__(self, output_dim):
        super(Logistic_Regression_2D, self).__init__()
        self.output_dim = output_dim
        self.linear = torch.nn.Linear(2, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        outputs = outputs.view(-1, self.output_dim)
        return outputs

class MNIST_1(nn.Module):
  def __init__(self, hidden_size, num_classes):
    super(MNIST_1, self).__init__()
    self.l1 = nn.Linear(28*28, hidden_size)
    self.relu = nn.ReLU()
    self.l2 = nn.Linear(hidden_size, num_classes)
  
  def forward(self, x):
    out = self.l1(x.reshape(-1, 28*28))
    out = self.relu(out)
    out = self.l2(out)
    return out

class MNIST_2(nn.Module):
  def __init__(self, num_classes):
    super(MNIST_2, self).__init__()
    self.l1 = nn.Linear(28*28, num_classes, bias=False)
  
  def forward(self, x):
    out = self.l1(x.reshape(-1, 28*28))
    return out

class MNIST_LogisticRegression(nn.Module):
    def __init__(self, num_classes):
        super(MNIST_LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(28*28, num_classes, bias=False)

    def forward(self, x):
        x = x.reshape(x.shape[0], 28*28)
        outputs = torch.sigmoid(self.linear(x))
        # outputs = self.linear(x)
        return outputs

class MNIST_Regression_2Layer(nn.Module):
    def __init__(self, hidden_size, num_classes):
        super(MNIST_Regression_2Layer, self).__init__()
        self.l1 = nn.Linear(28*28, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
  
    def forward(self, x):
        out = self.l1(x.reshape(-1, 28*28))
        out = self.relu(out)
        out = self.l2(out)
        out = torch.sigmoid(out)
        return out
