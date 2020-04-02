from model import ResidualBlock, ResNet
from train import train, infer
from utils import AccuracyMeter
from pieacewise_linear_lr_schedule import PiecewiseLinearLR #get_change_scale, get_piecewise
from transforms import TensorRandomHorizontalFlip, TensorRandomCrop, Cutout, Transform

from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
torch.set_default_tensor_type('torch.cuda.FloatTensor')

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import pickle
import datetime
import os
import logging
logging.basicConfig(level=logging.INFO)

import glob
# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    cifar10_mean, cifar10_std = [
    (125.31, 122.95, 113.87), # equals np.mean(cifar10()['train']['data'], axis=(0,1,2)) 
    (62.99, 62.09, 66.70), # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
    ]
    train_transform = [transforms.ToTensor(),
        # transforms.Normalize(cifar10_mean, cifar10_std),
        TensorRandomCrop(30, 30),
        TensorRandomHorizontalFlip(),
        Cutout(8, 8)
        ]
    
    test_transform =[transforms.ToTensor()
        ]

    batch_size = 8
    split = 0.2
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True)

    indices = list(range(int(split*len(trainset))))
    valid_indices =  list(range(int(split*len(trainset)), len(trainset)))
    logging.info(f"Training size= {len(indices)}")
    training_sampler = SubsetRandomSampler(indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    trainloader = torch.utils.data.DataLoader(dataset=Transform(trainset, train_transform),
                                            batch_size=batch_size,
                                            sampler=training_sampler) 

    validloader = torch.utils.data.DataLoader(dataset=Transform(trainset, test_transform), 
                                            batch_size=batch_size, 
                                            sampler=valid_sampler)                
    
    testloader = torch.utils.data.DataLoader(Transform(testset, test_transform),
                                            batch_size=batch_size,
                                            shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # cifar10_mean, cifar10_std = [
    # (125.31, 122.95, 113.87), # equals np.mean(cifar10()['train']['data'], axis=(0,1,2)) 
    # (62.99, 62.09, 66.70), # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
    # ]
    # #####################
    # ## data preprocessing
    # #####################
    # mean, std = [torch.tensor(x, device=device, dtype=torch.float16) for x in (cifar10_mean, cifar10_std)]

    # normalise = lambda data, mean=mean, std=std: (data - mean)/std
    # unnormalise = lambda data, mean=mean, std=std: data*std + mean
    # pad = lambda data, border: nn.ReflectionPad2d(border)(data)
    # transpose = lambda x, source='NHWC', target='NCHW': x.permute([source.index(d) for d in target]) 
    # to = lambda *args, **kwargs: (lambda x: x.to(*args, **kwargs))



    num_epochs = 35
    save_model_str = './models/'
    success = False
    time_to_94 = None
    if not os.path.exists(save_model_str):
        os.mkdir(save_model_str) 
    logging.info(f"{torch.cuda.is_available()}")
    cudnn.benchmark = True
    cudnn.enabled=True
    gpu = 'cuda:0'
    torch.cuda.set_device(gpu)
    model = ResNet(ResidualBlock, [1, 1, 1], initial_depth=64)
    model.cuda()
    # [1, 1, 1]
    # [2, 2, 2, 2]
    total_model_params = np.sum(p.numel() for p in model.parameters())
    # logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    
    save_model_str += f'model_({datetime.datetime.now()})'
    if not os.path.exists(save_model_str):
        os.mkdir(save_model_str) 
    criterion = nn.CrossEntropyLoss()
    criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.001) #), weight_decay=5e-4*batch_size, momentum=0.9)
    lr_scheduler = PiecewiseLinearLR(optimizer, milestones=[0, 5, num_epochs], schedule=[0, 0.4, 0])

    summary_dir = f'{save_model_str}/summary'
    if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)    
    c = datetime.datetime.now()
    train_meter = AccuracyMeter(model_dir=summary_dir, name='train')
    test_meter = AccuracyMeter(model_dir=summary_dir, name='test')
    valid_meter = AccuracyMeter(model_dir=summary_dir, name='valid')
    lrs = list()
    for epoch in range(num_epochs):
        lr = lr_scheduler.get_lr()[0]
        lrs.append(lr)

        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj, time = train(trainloader, model, criterion, optimizer)
        
        train_meter.update({'acc': train_acc, 'loss': train_obj}, time.total_seconds())
        lr_scheduler.step()
        valid_acc, valid_obj, time = infer(validloader, model, criterion)
        valid_meter.update({'acc': valid_acc, 'loss': valid_obj}, time.total_seconds())
        if valid_acc >=94:
            success = True
            time_to_94 = train_meter.time
            logging.info(f'Time to reach 94% {time_to_94}')    

    a = datetime.datetime.now() - c
    test_acc, test_obj, time = infer(testloader, model, criterion, type='test')
    test_meter.update({'acc': test_acc, 'loss': test_obj}, time.total_seconds())
    torch.save(model.state_dict(), f'{save_model_str}/state')
    train_meter.plot(save_model_str)
    valid_meter.plot(save_model_str)
    plt.plot(lrs)
    plt.title('LR vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.xticks(np.arange(0, num_epochs, 5))
    plt.savefig('lr_schedule.png')
    plt.close()

    total_time = round(a.total_seconds(), 2)
    logging.info(f'test_acc: {test_acc}, save_model_str:{save_model_str}, total time :{total_time} and GPU used {torch.cuda.get_device_name(0)}')
    _, cnt, time  = train_meter.get()
    time_per_step = round(time/cnt, 2)
    return_dict = {'test_acc': test_acc, 
                'save_model_str':save_model_str, 
                'training_time_per_step': time_per_step, 
                'total_train_time': time, 
                'total_time':total_time, 
                'GPU' :torch.cuda.get_device_name(0)
                }
    if success:
        return_dict['time_to_94'] = time_to_94
    return 


if __name__ == '__main__':
    ret_dict = main()
    with open('all_experiments.txt', 'a') as f:
        f.write(f"test accuracy:{ret_dict['test_acc']}, training_time_per_step: {ret_dict['training_time_per_step']}, total_train_time: {ret_dict['total_train_time']} , save_model_str:{ret_dict['save_model_str']}, and GPU used: {ret_dict['GPU']}\n")
        f.close()