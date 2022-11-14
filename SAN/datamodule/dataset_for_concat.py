from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch


class ConcatDataset(Dataset):
    def __init__(self, datasets, align=True):
        self.datasets = datasets
        self.transforms = transforms
        self.align = align

    def __getitem__(self, index):
        if self.align:
            data = [d[index] for d in self.datasets]
        else:
            data = [self.datasets[0][index]]
            for i,d in enumerate(self.datasets):
                if i == 0: continue
                id = torch.randint(0, len(d), (1,))
                data.append(d[id])
        return tuple(data)

    def __len__(self):
        return len(self.datasets[0])