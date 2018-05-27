import os
import PIL

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class camvid_data_loader(Dataset):
    def __init__(self):
        super(camvid_data_loader, self).__init__()

    def __getitem__(self, idx):
        raw = None
        label = None

        result = {'raw': raw, 'label': label}

        return result

    def __len__(self):
        return len()
