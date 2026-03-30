"""Dataset wrappers for serialized tensor datasets."""

import torch
from torch.utils.data import Dataset


class PairedECGDataset(Dataset):
    """Load a serialized paired-ECG tensor dataset from disk."""
    def __init__(self, path: str):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]


class ListDataset(Dataset):
    """Load a serialized flat tensor dataset from disk."""
    def __init__(self, path: str):
        self.data = torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        latent_dict = self.data[index]
        return latent_dict["data"], latent_dict["label"]
