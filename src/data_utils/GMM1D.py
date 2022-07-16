import os
import random
import numpy as np

import torch
from torch.utils.data import TensorDataset
import pandas as pd


def get_GMM1D_data():
    train_df = pd.read_csv("data/GMM1D/1D_1000sample_2class_train.csv")
    x = [eval(x) for x in train_df['x']]
    y = train_df['y']
    train_data = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))

    test_df = pd.read_csv("data/GMM1D/1D_1000sample_2class_test.csv")
    x = [eval(x) for x in test_df['x']]
    y = test_df['y']
    test_data = TensorDataset(torch.FloatTensor(x), torch.LongTensor(y))

    print(
        'loaded train_dataset with {} samples,test_dataset with {} samples, '.format(
            len(train_data),
            len(test_data)))

    return train_data, test_data
