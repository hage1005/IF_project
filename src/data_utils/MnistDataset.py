
## perform subsample and implement _get_data_raw(type) {"train", "dev"} ->
from src.data_utils.GeneralDataset import GeneralDataset 

import torch
import torchvision
from torchvision import transforms
from src.data_utils.FolderDataset import FolderDataset

class MnistDataset(GeneralDataset):

    class_label_dict = {
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

    class_map = [
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

    def __init__(self, dev_data_path, classes, num_per_class):
        label_transform_fn = lambda x: x
        image_transform_fn = transforms.Compose([ 
            transforms.Normalize(
            (0.1307,),(0.3081,))
            ])
        super().__init__(image_transform_fn, label_transform_fn)
        self.dev_data_path = dev_data_path
        self.num_per_class = num_per_class
        self.classes = classes

    def _get_data_raw(self, type):
        if type == 'train':
            return self._get_train_data_raw()
        elif type == 'dev':
            return self._get_dev_data_raw()
        else:
            raise NotImplementedError()

    def _get_train_data_raw(self): #subsample
        data = torchvision.datasets.MNIST(root='../data', train=True, download=True)
        t = transforms.ToTensor()
        train_data_raw_sampled = self._subsample_by_classes(
            data, self.classes, self.num_per_class)

        return torch.stack([t(x[0]) for x in train_data_raw_sampled]),\
            torch.stack([torch.tensor(x[1]) for x in train_data_raw_sampled])
    
    def _get_dev_data_raw(self):
        dev_data_no_transform = FolderDataset(self.dev_data_path)
        t = transforms.ToTensor()

        return torch.stack([t(x[0]) for x in dev_data_no_transform]),\
            torch.stack([torch.tensor(x[1]) for x in dev_data_no_transform])

    @classmethod
    def get_class_label_dict(self):
        return self.class_label_dict
    @classmethod
    def get_class_map(self):
        return self.class_map

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

    