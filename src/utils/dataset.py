import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd

class DataExample:
    def __init__(self, input, label):
        self.input = input
        self.label = label

def get_cifar10_data(classes, num_per_class):
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_data = torchvision.datasets.CIFAR10(root='../data', train=True,
                                            download=True, transform=transform)

    all_examples = [DataExample(input=t[0], label=t[1])
                    for t in train_data]

    train_data = _subsample_by_classes(all_examples, classes, num_per_class)

    test_data = torchvision.datasets.CIFAR10(root='../data', train=False,
                                            download=True, transform=transform)

    print('loaded train_dataset with {} samples,test_dataset with {} samples, '.format(len(train_data),len(test_data) ))
    
    return train_data, test_data

def get_GMM1D_data():
    train_df = pd.read_csv("data/1D_1000sample_2class_train.csv")
    x = [eval(x) for x in train_df['x']]
    y = train_df['y']
    train_data = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))

    test_df = pd.read_csv("data/1D_1000sample_2class_test.csv")
    x = [eval(x) for x in test_df['x']]
    y = test_df['y']
    test_data = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))

    print('loaded train_dataset with {} samples,test_dataset with {} samples, '.format(len(train_data),len(test_data) ))

    return train_data, test_data

def get_GMM2D_data():
    train_df = pd.read_csv("data/2D_1000sample_2class_train.csv")
    x = [eval(x) for x in train_df['x']]
    y = train_df['y']
    train_data = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))

    test_df = pd.read_csv("data/2D_1000sample_2class_test.csv")
    x = [eval(x) for x in test_df['x']]
    y = test_df['y']
    test_data = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))

    print('loaded train_dataset with {} samples,test_dataset with {} samples, '.format(len(train_data),len(test_data) ))

    return train_data, test_data

def _subsample_by_classes(all_examples, labels, num_per_class=None):
    if num_per_class is None:
        return all_examples

    examples = {label: [] for label in labels}
    for example in all_examples:
        if example.label in labels:
            examples[example.label].append(example)

    picked_examples = []
    for label in labels:
        random.shuffle(examples[label])

        examples_with_label = examples[label][:num_per_class[label]]
        picked_examples.extend(examples_with_label)

        print(f'number of examples with label \'{label}\': '
              f'{len(examples_with_label)}')

    return picked_examples