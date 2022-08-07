
from src.data_utils.ImageDataset import ImageDataset 

import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision import transforms
from src.data_utils.FolderDataset import FolderDataset

class Cifar10Dataset(ImageDataset):
            
    def get_train(self):
        trans = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

        train_data_no_transform_raw = torchvision.datasets.CIFAR10(
            root='../data', train=True, download=True, transform=transforms.ToTensor())

        train_data_no_transform_sampled = self._subsample_by_classes(
            train_data_no_transform_raw, self.classes, self.num_per_class)

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

    