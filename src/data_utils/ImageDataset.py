import random

from src.data_utils.FolderDataset import FolderDataset
class ImageDataset:
    def __init__(self, dev_original_dir, dev_transformed_dir,test_original_dir , test_transformed_dir, classes, num_per_class=None):
        self.dev_transformed_dir = dev_transformed_dir
        self.dev_original_dir = dev_original_dir
        self.test_transformed_dir = test_transformed_dir
        self.test_original_dir = test_original_dir
        self.classes = classes
        self.num_per_class = num_per_class

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
    
    def get_dev(self):
        dev_data = FolderDataset(self.dev_transformed_dir)
        dev_data_no_transform = FolderDataset(self.dev_original_dir)
        print('loaded dev FolderDataset with {} files in folder, '.format(len(dev_data)))

        return dev_data, dev_data_no_transform
    
    def get_test(self):
        test_data = FolderDataset(self.test_transformed_dir)
        test_data_no_transform = FolderDataset(self.test_original_dir)
        print('loaded test FolderDataset with {} files in folder, '.format(len(test_data)))

        return test_data, test_data_no_transform
