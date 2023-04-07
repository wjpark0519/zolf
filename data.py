import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, y_data, aod_data, aoa_data):
        self.y_data = y_data
        self.aoa_data = aoa_data
        self.aod_data = aod_data

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        y = torch.FloatTensor(self.y_data[idx])
        aoa = torch.FloatTensor(self.aoa_data[idx])
        aod = torch.FloatTensor(self.aod_data[idx])
        return y, aod, aoa
