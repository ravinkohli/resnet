from model import ResidualBlock, ResNet
from train import train, infer
from utils import AccuracyMeter
from pieacewise_linear_lr_schedule import PiecewiseLinearLR #get_change_scale, get_piecewise
from transforms import TensorRandomHorizontalFlip, TensorRandomCrop, Cutout

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

if __name__ == '__main__':
    cifar10_mean, cifar10_std = [
    (125.31, 122.95, 113.87), # equals np.mean(cifar10()['train']['data'], axis=(0,1,2)) 
    (62.99, 62.09, 66.70), # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
    ]
    train_transform = transforms.Compose(
        [transforms.ToTensor(),
        # transforms.Normalize(cifar10_mean, cifar10_std),
        TensorRandomCrop(30, 30),
        TensorRandomHorizontalFlip(),
        Cutout(8, 8),
        ])
    
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        ])

    batch_size = 8
    split = 0.2
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    indices = list(range(int(split*len(trainset))))
    valid_indices =  list(range(int(split*len(trainset)), len(trainset)))
    logging.info(f"Training size= {len(indices)}")
    training_sampler = SubsetRandomSampler(indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    trainloader = torch.utils.data.DataLoader(dataset=trainset,
                                            batch_size=batch_size,
                                            sampler=training_sampler) 

    validloader = torch.utils.data.DataLoader(dataset=trainset, 
                                            batch_size=batch_size, 
                                            sampler=valid_sampler)                
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
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
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    # lr_scheduler = get_change_scale(
    #     get_piecewise([0, 4, num_epochs], [0.025, 0.4, 0.001]),
    #     1.0 / batch_size
    # )
    summary_dir = f'{save_model_str}/summary'
    if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)    
    c = datetime.datetime.now()
    train_meter = AccuracyMeter(model_dir=summary_dir, name='train')
    test_meter = AccuracyMeter(model_dir=summary_dir, name='test')
    valid_meter = AccuracyMeter(model_dir=summary_dir, name='valid')
    for epoch in range(num_epochs):
        lr = lr_scheduler.get_lr()[0]
        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj, time = train(trainloader, model, criterion, optimizer)
        
        train_meter.update({'acc': train_acc, 'loss': train_obj}, time.total_seconds())
        lr_scheduler.step()
        valid_acc, valid_obj, time = infer(validloader, model, criterion)
        valid_meter.update({'acc': valid_acc, 'loss': valid_obj}, time.total_seconds())
        if valid_acc >=94:
            logging.info(f'Time to reach 94% {train_meter.time}')    

    a = datetime.datetime.now() - c
    test_acc, test_obj, time = infer(testloader, model, criterion, type='test')
    test_meter.update({'acc': test_acc, 'loss': test_obj}, time.total_seconds())
    torch.save(model.state_dict(), f'{save_model_str}/state')
    train_meter.plot(save_model_str)
    valid_meter.plot(save_model_str)
    # if save_model_str:
    #     # Save the model checkpoint, can be restored via "model = torch.load(save_model_str)"
    
        
    #     # get some random training images
    #     model.eval()
    #     dataiter = iter(trainloader)
    #     images, labels = dataiter.next()
    #     images = images.cuda()
    #     labels = labels.cuda(non_blocking=True)
    #     # create grid of images
    #     # img_grid = torchvision.utils.make_grid(images)

    #     # show images
    #     # matplotlib_imshow(img_grid, one_channel=True)

    #     # write to tensorboard
    #     writer.add_graph(model, images)
    #     # images, labels = select_n_random(trainset.data, trainset.targets)

    #     # # get the class labels for each image
    #     # class_labels = [classes[lab] for lab in labels]

    #     # # log embeddings
    #     # features = images.view(-1, 28 * 28)
    #     # writer.add_embedding(features,
    #     #                     metadata=class_labels,
    #     #                     label_img=images.unsqueeze(1))
    #     writer.close()

    logging.info(f'test_acc: {test_acc}, save_model_str:{save_model_str}, total time :{a.total_seconds()} and GPU used {torch.cuda.get_device_name(0)}')



