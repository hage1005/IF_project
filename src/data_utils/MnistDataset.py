
from src.data_utils.ImageDataset import ImageDataset 

import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision import transforms
from src.data_utils.FolderDataset import FolderDataset

class MnistDataset(ImageDataset):
    
    def get_train_data_no_transform_raw(self):
        return torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=None)


    @classmethod
    def get_class_label_dict(self):
        return {
            '0': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '4': 4,
            '5': 5,
            '6': 6,
            '7': 7,
            '8': 8,
            '9': 9,
        }
    @classmethod
    def get_class_map(self):
        return [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5',
            '6',
            '7',
            '8',
            '9',
        ]


    