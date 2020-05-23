"""
from kakao dawnbench submission, https://github.com/wbaek/torchskeleton/releases/tag/v0.2.1_dawnbench_cifar10_release and dacid c pahe
"""
import torch
from torch import nn
import numpy as np
import torchvision
from settings import get
class Cutout:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        h, w = image.size(1), image.size(2)
        y = np.random.choice(range(h))
        x = np.random.choice(range(w))
        image[..., y:y+self.height, x:x+self.width] = 0.0
        return image

    def __repr__(self):
        return self.__class__.__name__ + '(height={0}, width={1})'.format(self.height, self.width)

class TensorRandomHorizontalFlip:
    def __call__(self, tensor):
        choice = np.random.choice([True, False])
        # return tensor[..., ::-1].copy() if choice else tensor
        return torch.flip(tensor, dims=[-1]) if choice else tensor


class TensorRandomCrop:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, tensor):
        C, H, W = tensor.shape
        h = np.random.choice(range(H + 1 - self.height))
        w = np.random.choice(range(W + 1 - self.width))
        return tensor[:, h:h+self.height, w:w+self.width]

class Transpose:
    def __init__(self, source, target):
        self.source = source
        self.target = target
    
    def __call__(self, x):
        return x.permute([self.source.index(d) for d in self.target]) 
        # return x.transpose([self.source.index(d) for d in self.target]) 

# class Transpose:
#     def __init__(self, source, target):
#         self.source = source
#         self.target = target
    
#     def __call__(self, x):

# # 

# class Pad:
#     def __init__(self, border):
#         self.border = border
    
#     def __call__(self, tensor):

class Pad:
    def __init__(self, border):
        self.border = border
    
    def __call__(self, x):
        # return np.pad(x, [(0, 0), (self.border, self.border), (self.border, self.border), (0, 0)], mode='reflect')
        return nn.ReflectionPad2d(self.border)(x)


class Normalise:
    def __init__(self, mean, std):
        if config['device'] == 'cpu':
            self.mean, self.std = [torch.tensor(x, dtype=get('dtype')).cuda() for x in (mean, std)]
        else:
            self.mean, self.std = [torch.tensor(x, dtype=get('dtype')).cuda() for x in (mean, std)]
    def __call__(self, x):
        return (x - self.mean)/self.std

class To:
    def __init__(self, arg):
        self.arg = arg
    
    def __call__(self, x):
        return torch.tensor(x, dtype=self.arg)

class Transform():
    def __init__(self, dataset, transforms):
        self.dataset, self.transforms = dataset, transforms
        
    def __len__(self):
        return len(self.dataset)
           
    def __getitem__(self, index):
        data, labels = self.dataset[index]
        data = data.copy()
        for f in self.transforms:
            data = f(data)
        return data, labels
    
