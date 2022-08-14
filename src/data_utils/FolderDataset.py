import torch
import os
from torch.utils.data import Dataset
from torchvision import transforms
class FolderDataset(Dataset):
    def __init__(self, folder, trans = None):
        self.folder = folder
        self.trans = trans

    def __len__(self):
        return len(os.listdir(self.folder))

    def __getitem__(self, idx):
        assert(os.path.exists(self.folder))
        try:
            res = torch.load(f"{self.folder}/tensor{idx}.pt")
            if self.trans:
                res = ( self.trans(res[0]), res[1]) #this automatically scale 0-255 to 0-1
            return res
        except BaseException:
            raise IndexError()

    