import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
class FolderDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder

    def __len__(self):
        return len(os.listdir(self.folder))

    def __getitem__(self, idx):
        try:
            res = torch.load(f"{self.folder}/tensor{idx}.pt")
            if not torch.is_tensor(res[0]):
                res[0] = transforms.ToTensor()(res[0]) #this automatically scale 0-255 to 0-1
            return res
        except BaseException:
            raise IndexError()

    