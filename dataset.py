import numpy as np
import torch
import copy

chunks = lambda data, splits: (data[start:end] for (start, end) in zip(splits, splits[1:]))

class DataLoader():
    def __init__(self, dataset, batch_size, transforms=None, shuffle=True):
        self.dataset = dataset
        self.transforms = transforms
        self.shuffle = shuffle
        self.splits = np.arange(0, len(self.dataset)+1, batch_size)

    def __iter__(self):
        data, targets = self.dataset.data.clone().detach(), torch.tensor(self.dataset.targets)
        if self.transforms is not None:
            for transform in self.transforms:
                data = torch.stack([transform(x) for x in data])
        if self.shuffle:
            i = torch.randperm(len(data))
            data, targets = data[i], targets[i]
        return ((x.clone(), y) for (x, y) in zip(chunks(data, self.splits), chunks(targets, self.splits)))

    def __len__(self):
        return len(self.splits) - 1