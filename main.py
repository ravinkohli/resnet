from model import ResNet, build_network, ResidualBlock, Network
from train import train, infer
from utils import AccuracyMeter, write_to_file
from pieacewise_linear_lr_schedule import PiecewiseLinearLR #get_change_scale, get_piecewise
import transform
from torch_backend import BatchNorm
from settings import get_dict
from torch import nn
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import SubsetRandomSampler
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.autograd.set_detect_anomaly(True)
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import wandb


import pickle
import datetime
import os
import logging
logging.basicConfig(level=logging.INFO)
import skeleton
import glob
import sys

def model_train(model, config, criterion, trainloader, testloader, validloader, model_name):
    num_epochs = config['budget']
    success = False
    time_to_94 = None
    logging.info(f"{torch.cuda.is_available()}")
    
    cudnn.benchmark = True
    cudnn.enabled=True
    gpu = 'cuda:0'
    torch.cuda.set_device(gpu)
    lrs = list()
    print(f"weight decay:\t{config['weight_decay']}")
    print(f"momentum :\t{config['momentum']}")

    optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=config['weight_decay'], momentum=config['momentum'])
    lr_scheduler = PiecewiseLinearLR(optimizer, milestones=config['milestones'], schedule=config['schedule'])
    save_model_str = './models/'
    
    if not os.path.exists(save_model_str):
        os.mkdir(save_model_str) 
   
    save_model_str += f'model_({datetime.datetime.now()})'
    if not os.path.exists(save_model_str):
        os.mkdir(save_model_str) 
    
    summary_dir = f'{save_model_str}/summary'
    if not os.path.exists(summary_dir):
            os.mkdir(summary_dir)    
    c = datetime.datetime.now()
    train_meter = AccuracyMeter(model_dir=summary_dir, name='train')
    test_meter = AccuracyMeter(model_dir=summary_dir, name='test')
    valid_meter = AccuracyMeter(model_dir=summary_dir, name='valid')
    
    for epoch in range(num_epochs):
        lr = lr_scheduler.get_lr()[0]
        lrs.append(lr)

        logging.info('epoch %d lr %e', epoch, lr)

        train_acc, train_obj, time = train(trainloader, model, criterion, optimizer, name=model_name)
        
        train_meter.update({'acc': train_acc, 'loss': train_obj}, time.total_seconds())
        lr_scheduler.step()
        valid_acc, valid_obj, time = infer(testloader, model, criterion, name=model_name)
        valid_meter.update({'acc': valid_acc, 'loss': valid_obj}, time.total_seconds())
        if valid_acc >=94:
            success = True
            time_to_94 = train_meter.time
            logging.info(f'Time to reach 94% {time_to_94}')  
            wandb.log({
        "Test Accuracy":valid_acc,
        "Test Loss": valid_obj,
        "Train Accuracy":train_acc,
        "Train Loss": train_obj})  

    a = datetime.datetime.now() - c
    test_acc, test_obj, time = infer(testloader, model, criterion, type='test', name=model_name)
    test_meter.update({'acc': test_acc, 'loss': test_obj}, time.total_seconds())
    torch.save(model.state_dict(), f'{save_model_str}/state')
    wandb.save('model.h5')
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
    return return_dict

def get_skeleton_model(criterion):
    model = build_network()
    model.cuda()
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            if hasattr(module, 'weight') and module.weight is not None:
                module.weight.data.fill_(1.0)
            module.eps = 0.00001
            module.momentum = 0.1
        else:
            module.half()
        if isinstance(module, torch.nn.Conv2d) and hasattr(module, 'weight'):
            # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
            # torch.nn.init.xavier_uniform_(module.weight, gain=torch.nn.init.calculate_gain('linear'))
        if isinstance(module, torch.nn.Linear) and hasattr(module, 'weight'):
            # torch.nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))  # original
            torch.nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='linear')
            # torch.nn.init.xavier_uniform_(module.weight, gain=1.)
    class ModelLoss(torch.nn.Module):
        def __init__(self, model, criterion):
            super(ModelLoss, self).__init__()
            self.model = model
            self.criterion = criterion

        def forward(self, inputs, targets):
            inputs = inputs.cuda().to(dtype=torch.half)
            targets = targets.cuda(non_blocking=True)#.to(dtype=torch.half)
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)
            return logits, loss
    model = ModelLoss(model, criterion)
    return model
    
def main(config):
    cifar10_mean, cifar10_std = [
    (125.31, 122.95, 113.87), # equals np.mean(cifar10()['train']['data'], axis=(0,1,2)) 
    (62.99, 62.09, 66.70), # equals np.std(cifar10()['train']['data'], axis=(0,1,2))
    ]

    torch.manual_seed(config['seed'])  
    train_transform = [
        # transform.Pad(2),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transform.TensorRandomCrop(32, 32),
        transform.TensorRandomHorizontalFlip(),
        transform.Cutout(8, 8)
        ]

    test_transform =[
        # transform.Pad(2),
        transforms.ToTensor()
        # transform.Transpose(source='NHWC', target='NCHW'),
        ]


    batch_size = config['batch_size']

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True)

    logging.info(f"Training size= {len(trainset)}")
    trans_trainset = transform.Transform(trainset, train_transform)
    trainloader = torch.utils.data.DataLoader(dataset=trans_trainset,
                                            batch_size=batch_size, shuffle=False)       

    testloader = torch.utils.data.DataLoader(transform.Transform(testset, test_transform),
                                            batch_size=batch_size,
                                            shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = nn.CrossEntropyLoss(reduction='sum')
    criterion.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    model_name = config['model']
    # steps_per_epoch = int(steps_per_epoch * 1.0)
    logging.info(f"Model:{model_name}")
    if model_name =='skeleton':
        criterion = nn.CrossEntropyLoss(reduction='sum')
        criterion.cuda()
        model = get_skeleton_model(criterion)
        model.cuda()
    elif model_name == 'self-vanilla':
        model = ResNet(ResidualBlock, [1, 1, 1], initial_depth=64, batch_norm=config['batch_norm'])
        model.cuda()
    elif model_name == 'self-david':
        model = Network(batch_norm=config['batch_norm'])
        model.cuda()
    else:
        logging.error('incorrect model')
        sys.exit()
    wandb.init(entity='wandb', project='resnet-dawnbench')
    wandb.watch_called = False
    wandb.watch(model, log="all")
    ret_dict = model_train(model, config, criterion, trainloader, testloader, testloader, model_name)

    file_name = "experiments.txt"
    write_to_file(ret_dict, file_name)
    write_to_file(config, file_name)


if __name__ == '__main__':
    settings = get_dict()
    config = dict()
    config['batch_size'] = settings['batch_size']
    config['budget'] = settings['budget']
    config['model'] = settings['name']
    config['weight_decay'] =  0 #5e-4*config['batch_size']
    config['momentum'] = 0 #.9
    config['milestones'] = [0, 4, config['budget']]
    config['schedule'] = [0, 0.4, 0]
    config['batch_norm'] = BatchNorm
    config['seed'] = 42
    main(config)