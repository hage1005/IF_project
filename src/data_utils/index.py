
from .MnistDataset import MnistDataset
from .Cifar10Dataset import Cifar10Dataset
from .GeneralDataset import GeneralDataset
from typing import Union

def get_dataset(name) -> Union[GeneralDataset, str]:
    if name == 'cifar10':
        return Cifar10Dataset, "image"
    elif name == 'mnist':
        return MnistDataset, "image"
    else:
        raise NotImplementedError()