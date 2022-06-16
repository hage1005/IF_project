import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

def return_data(batch_size, dataset_name):
    if dataset_name == 'cifar10':
        num_workers = 2
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_data = torchvision.datasets.CIFAR10(root='../data', train=True,
                                                download=True, transform=transform)
        train_loader = DataLoader(train_data, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)

        test_data = torchvision.datasets.CIFAR10(root='../data', train=False,
                                                download=True, transform=transform)
        test_loader = DataLoader(test_data, batch_size=batch_size,
                                shuffle=True, num_workers=num_workers)

        print('loaded train_dataset with {} samples,test_dataset with {} samples, '.format(len(train_data),len(test_data) ))
        
        return train_loader, test_loader
    elif dataset_name == "GMM1D":
        train_df = pd.read_csv("../data/1D_1000sample_2class_train.csv")
        train_data = TensorDataset(torch.tensor(train_df['x']).reshape(-1,1), torch.tensor(train_df['y']).reshape(-1,1))
        train_loader = DataLoader(train_data, batch_size=batch_size,
                                shuffle=True, num_workers=1)

        test_df = pd.read_csv("../data/1D_1000sample_2class_test.csv")
        test_data = TensorDataset(torch.tensor(test_df['x']).reshape(-1,1), torch.tensor(test_df['y']))
        test_loader = DataLoader(test_data, batch_size=batch_size,
                                shuffle=True, num_workers=1)

        print('loaded train_dataset with {} samples,test_dataset with {} samples, '.format(len(train_data),len(test_data) ))

        return train_loader, test_loader