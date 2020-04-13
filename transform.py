"""
from kakao dawnbench submission, https://github.com/wbaek/torchskeleton/releases/tag/v0.2.1_dawnbench_cifar10_release and dacid c pahe
"""
import torch
import numpy as np
import torchvision

class Cutout:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, image):
        h, w = image.size(1), image.size(2)
        # mask = np.ones((h, w), np.float32)
        y = np.random.choice(range(h))
        x = np.random.choice(range(w))
        image[..., y:y+self.height, x:x+self.width] = 0.0
        # y1 = np.clip(y - self.height // 2, 0, h)
        # y2 = np.clip(y + self.height // 2, 0, h)
        # x1 = np.clip(x - self.width // 2, 0, w)
        # x2 = np.clip(x + self.width // 2, 0, w)

        # mask[y1: y2, x1: x2] = 0.
        # mask = torch.from_numpy(mask).to(device=image.device, dtype=image.dtype)
        # mask = mask.expand_as(image)
        # image *= mask
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
    
    def __call__(self, tensor):
        return tensor.permute([self.source.index(d) for d in self.target]) 

class Pad:
    def __init__(self, padding):
        self.padding = padding
    
    def __call__(self, tensor):
        return torchvision.transforms.functional.pad(tensor, self.padding, padding_mode='reflect')



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
    