import torch
from torch.utils.data import TensorDataset


## get_data() : string type -> TensorDataset, apply transform

class GeneralDataset:

    def __init__(self, data_transfrom_fn, label_transform_fn):
        self.data_transfrom_fn = data_transfrom_fn
        self.label_transform_fn = label_transform_fn
    

    def get_data(self, type, transform=True):
        data_raw, label_raw = self._get_data_raw(type)

        if transform:
            data = self._transform_data(data_raw, self.data_transfrom_fn)
            label = self._transform_data(label_raw, self.label_transform_fn)
        else:
            data = data_raw
            label = label_raw

        return TensorDataset(data, label)
    
    def _get_data_raw(self, type):
        pass ## to be implemented by subclass
    
    def _transform_data(self, data_raw, transform):
        return torch.stack([transform(x) for x in data_raw])
    
    
