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
        # print(((x.clone(), y) for (x, y) in zip(chunks(data, self.splits), chunks(targets, self.splits))))
        return ((x.clone(), y) for (x, y) in zip(chunks(data, self.splits), chunks(targets, self.splits)))

    def __len__(self):
        return len(self.splits) - 1


'''
Code for prefetching modified from 
https://gist.githubusercontent.com/xhchrn/45585e33c4f1f18864309221eda2f046/raw/0a1feca64ffef2e9390be1b750583208a01d4172/data_prefetcher.py
'''

class DataPrefetchLoader():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_data, self.next_target = next(self.loader)
        except StopIteration:
            self.next_data = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_data = self.next_data.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
                    
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        data, targets = self.next_data, self.next_target
        if data is None:
            raise StopIteration
        self.preload()
        return data, targets
    
    def __iter__(self):
        return self
    
    def __len__(self): 
        return len(self.dataloader)

