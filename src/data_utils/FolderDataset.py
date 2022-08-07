import torch
import os
from torch.utils.data import Dataset

class FolderDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder

    def __len__(self):
        return len(os.listdir(self.folder))

    def __getitem__(self, idx):
        try:
            return torch.load(f"{self.folder}/tensor{idx}.pt")
        except BaseException:
            raise IndexError()

    