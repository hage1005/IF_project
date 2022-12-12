import torch
from torch.utils.data import TensorDataset



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
    
    @classmethod
    def get_class_label_dict(self):
        pass ## to be implemented by subclass
    @classmethod
    def get_class_map(self):
        pass ## to be implemented by subclass

    def _subsample_by_classes(self, all_examples, labels, num_per_class=None):
        if num_per_class is None:
            return all_examples

        examples = {label: [] for label in labels}
        for example in all_examples:
            if example[1] in labels:
                examples[example[1]].append(example)

        picked_examples = []
        for label in labels:
            examples_with_label = examples[label][:num_per_class[label]]
            picked_examples.extend(examples_with_label)

            print(f'number of examples with label \'{label}\': '
                f'{len(examples_with_label)}')

        return picked_examples
    
    

## get_data() : string type -> TensorDataset, apply transform