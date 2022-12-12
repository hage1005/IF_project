import os
import random
import numpy as np

import torch
from torch.utils.data import TensorDataset
from .GeneralDataset import GeneralDataset
import pandas as pd

class GMM2DDataset(GeneralDataset):

    class_label_dict = {
        '0': 0,
        '1': 1
    }

    class_map = [
        '0',
        '1'
    ]

    def __init__(self, classes, num_per_class):
        label_transform_fn = lambda x: x
        data_transform_fn = lambda x: x
        super().__init__(data_transform_fn, label_transform_fn)
        self.num_per_class = num_per_class
        self.classes = classes

    def _get_data_raw(self, type):
        if type == 'train':
            df = pd.read_csv("data/GMM2D/2D_1000sample_2class_train.csv")
        elif type == 'dev':
            df = pd.read_csv("data/GMM2D/2D_1000sample_2class_test.csv")
        else:
            raise NotImplementedError()

        x = [eval(x) for x in df['x']]
        y = df['y']
        data_sampled = self._subsample_by_classes(zip(x, y), self.classes, self.num_per_class)
        x = [x[0] for x in data_sampled]
        y = [x[1] for x in data_sampled]

        return torch.stack([torch.FloatTensor(x[0]) for x in data_sampled]),\
            torch.stack([torch.tensor(x[1]) for x in data_sampled])

    @classmethod
    def get_class_label_dict(self):
        return self.class_label_dict
    @classmethod
    def get_class_map(self):
        return self.class_map
