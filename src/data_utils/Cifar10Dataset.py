
from src.data_utils.GeneralDataset import GeneralDataset 

import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision import transforms
from src.data_utils.FolderDataset import FolderDataset

class Cifar10Dataset(GeneralDataset):

    
    def __init__(self, dev_data_path, classes, num_per_class):
        label_transform_fn = lambda x: torch.tensor(x)
        image_tranform_fn = transforms.Compose([ 
            transforms.ToTensor(), 
            transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        super().__init__(image_tranform_fn, label_transform_fn)
        self.dev_data_path = dev_data_path
        self.num_per_class = num_per_class
        self.classes = classes
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

    