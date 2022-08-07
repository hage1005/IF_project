
from src.data_utils.ImageDataset import ImageDataset 

import torch
import torchvision
from torch.utils.data import TensorDataset
from torchvision import transforms
from src.data_utils.FolderDataset import FolderDataset

class MnistDataset(ImageDataset):
            
    def get_train(self):
        trans = transforms.Normalize(
            (0.1307,),(0.3081,))

        train_data_no_transform_raw = torchvision.datasets.MNIST(
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


    