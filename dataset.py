# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader


class HFSDataset(Dataset):
    def __init__(self, x_ecg, x_tab, y):
        self.x_ecg = torch.FloatTensor(x_ecg).permute(0, 2, 1)  # (N, 12, L)
        self.x_tab = torch.FloatTensor(x_tab)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_ecg[idx], self.x_tab[idx], self.y[idx]


def get_loader(x_ecg, x_tab, y, batch_size=64, shuffle=True):
    ds = HFSDataset(x_ecg, x_tab, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)
