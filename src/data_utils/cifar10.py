import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, TensorDataset
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd


class FolderDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder

    def __len__(self):
        return len(os.listdir(self.folder))

    def __getitem__(self, idx):
        try:
            return torch.load(f"{self.folder}/tensor{idx}.pt")
        except BaseException:
            raise IndexError()


def get_cifar10_train(classes, num_per_class=None):
    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994,
    # 0.2010))])
    trans = transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    train_data_no_transform_raw = torchvision.datasets.CIFAR10(
        root='../data', train=True, download=True, transform=transforms.ToTensor())

    train_data_no_transform_sampled = _subsample_by_classes(
        train_data_no_transform_raw, classes, num_per_class)

    train_data_x_no_transform = torch.stack(
        [example[0] for example in train_data_no_transform_sampled])
    train_data_x_transformed = torch.stack(
        [trans(x) for x in train_data_x_no_transform])
    train_data_y = torch.LongTensor(
        [example[1] for example in train_data_no_transform_sampled])

    assert(len(train_data_x_transformed) == len(train_data_x_no_transform))

    train_data_tensor = TensorDataset(train_data_x_transformed, train_data_y)
    train_data_no_transform_tensor = TensorDataset(
        train_data_x_no_transform, train_data_y)

    print('loaded train_dataset with {} samples'.format(
        len(train_data_no_transform_sampled)))

    return train_data_tensor, train_data_no_transform_tensor


def get_cifar10_test():
    test_data = FolderDataset("data/cifar10/test/transformed_sampled")
    test_data_no_transform = FolderDataset(
        "data/cifar10/test/original_sampled")
    print('loaded test_dataset with {} samples, '.format(len(test_data)))

    return test_data, test_data_no_transform


def _subsample_by_classes(all_examples, labels, num_per_class=None):
    if num_per_class is None:
        return all_examples

    examples = {label: [] for label in labels}
    for example in all_examples:
        if example[1] in labels:
            examples[example[1]].append(example)

    picked_examples = []
    for label in labels:
        random.shuffle(examples[label])

        examples_with_label = examples[label][:num_per_class[label]]
        picked_examples.extend(examples_with_label)

        print(f'number of examples with label \'{label}\': '
              f'{len(examples_with_label)}')

    return picked_examples
