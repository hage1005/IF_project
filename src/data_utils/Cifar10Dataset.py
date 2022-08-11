
from src.data_utils.ImageDataset import ImageDataset 

import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision import transforms
from src.data_utils.FolderDataset import FolderDataset

class Cifar10Dataset(ImageDataset):
        
    def get_train_data_no_transform_raw(self):
        return torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=None)
        
    @classmethod
    def get_class_label_dict(self):
        return {
            'airplane': 0,
            'automobile': 1,
            'bird': 2,
            'cat': 3,
            'deer': 4,
            'dog': 5,
            'frog': 6,
            'horse': 7,
            'ship': 8,
            'truck': 9,
        }   
    @classmethod
    def get_class_map(self):
        return [
            'airplane',
            'automobile',
            'bird',
            'cat',
            'deer',
            'dog',
            'frog',
            'horse',
            'ship',
            'truck',
        ]

    