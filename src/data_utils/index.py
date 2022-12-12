
from .MnistDataset import MnistDataset
from .Cifar10Dataset import Cifar10Dataset
from .GMM2D import GMM2DDataset
from .GeneralDataset import GeneralDataset
from typing import Union

def get_dataset(name) -> Union[GeneralDataset, str]:
    if name == 'cifar10':
        return Cifar10Dataset, "image"
    elif name == 'mnist':
        return MnistDataset, "image"
    elif name == 'GMM2D':
        return GMM2DDataset, "GMM2D"
    else:
        raise NotImplementedError()