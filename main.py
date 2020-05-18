from model import ResNet, build_network, ResidualBlock, Network, conv_bn_act_pool, conv_bn_pool_act,  conv_pool_bn_act
from train import train, infer
from utils import AccuracyMeter, write_to_file, count_parameters_in_MB, weights_init_uniform, preprocess
from lr_schedulers import PiecewiseLinearLR, SWAResNetLR #get_change_scale, get_piecewise
import transform
from torch_backend import BatchNorm, GhostBatchNorm 
from criterion import LabelSmoothLoss, NMTCriterion
from dataset import DataLoader
from settings import get_dict
from torch import nn

import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchcontrib
torch.set_default_tensor_type('torch.cuda.FloatTensor')

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
from functools import partial

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
    logging.info(f"weight decay:\t{config['weight_decay']}")
    logging.info(f"momentum :\t{config['momentum']}")

    base_optimizer = optim.SGD(model.parameters(), lr=config['base_lr'], weight_decay=config['weight_decay'], momentum=config['momentum'])
    if config['swa']:
        optimizer = torchcontrib.optim.SWA(base_optimizer)
        
        # lr_scheduler = SWAResNetLR(optimizer, milestones=config['milestones'], schedule=config['schedule'], swa_start=config['swa_start'], swa_init_lr=config['swa_init_lr'], swa_step=config['swa_step'], base_lr=config['base_lr'])
    else:
        optimizer = base_optimizer
        # lr_scheduler = PiecewiseLinearLR(optimizer, milestones=config['milestones'], schedule=config['schedule'])

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    #lr_scheduler = PiecewiseLinearLR(optimizer, milestones=config['milestones'], schedule=config['schedule'])
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

        logging.info('epoch %d, lr %e', epoch, lr)
        
        train_acc, train_obj, time = train(trainloader, model, criterion, optimizer, model_name, config['grad_clip'], config['prefetch'])
        
        train_meter.update({'acc': train_acc, 'loss': train_obj}, time.total_seconds())
        lr_scheduler.step()
        if config['swa'] and ((epoch+1) >= config['swa_start']) and ((epoch+1 - config['swa_start']) % config['swa_step'] == 0):
            optimizer.update_swa()
        valid_acc, valid_obj, time = infer(testloader, model, criterion, name=model_name,  prefetch=config['prefetch'])
        valid_meter.update({'acc': valid_acc, 'loss': valid_obj}, time.total_seconds())
        if valid_acc >=94:
            success = True
            time_to_94 = train_meter.time
            logging.info(f'Time to reach 94% {time_to_94}')  
        # wandb.log({"Test Accuracy":valid_acc, "Test Loss": valid_obj, "Train Accuracy":train_acc, "Train Loss": train_obj})  

    a = datetime.datetime.now() - c
    if config['swa']:
        optimizer.swap_swa_sgd()
        optimizer.bn_update(trainloader, model)
    test_acc, test_obj, time = infer(testloader, model, criterion, name=model_name, prefetch=config['prefetch'])
    test_meter.update({'acc': test_acc, 'loss': test_obj}, time.total_seconds())
    torch.save(model.state_dict(), f'{save_model_str}/state')
    # wandb.save('model.h5')
    train_meter.plot(save_model_str)
    valid_meter.plot(save_model_str)

    plt.plot(lrs)
    plt.title('LR vs epochs')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.xticks(np.arange(0, num_epochs, 5))
    plt.savefig(f'{save_model_str}/lr_schedule.png')
    plt.close()

    total_time = round(a.total_seconds(), 2)
    logging.info(f'test_acc: {test_acc}, save_model_str:{save_model_str}, total time :{total_time} and GPU used {torch.cuda.get_device_name(0)}')
    _, cnt, time  = train_meter.get()
    time_per_step = round(time/cnt, 2)
    return_dict = {
                'test_acc': test_acc, 
                'save_model_str':save_model_str, 
                'training_time_per_step': time_per_step, 
                'total_train_time': time, 
                'total_time':total_time, 
                'GPU' :torch.cuda.get_device_name(0),
                'train_acc': train_acc
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
    if config['half']:
        settings['dtype'] = torch.float16
    else:
        settings['dtype'] = torch.float32
    
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]
    torch.manual_seed(config['seed'])  
    preprocess_train_transforms = [
        transform.Pad(4),
        transform.Transpose(source='NHWC', target='NCHW'),
        transform.Normalise(CIFAR_MEAN, CIFAR_STD),
        transform.To(settings['dtype']),
    ]

    train_transforms = [
        transform.TensorRandomCrop(32, 32),
        transform.TensorRandomHorizontalFlip(),
        transform.Cutout(8, 8)
        ]

    preprocess_test_transforms =[
        transform.Transpose(source='NHWC', target='NCHW'),
        transform.Normalise(CIFAR_MEAN, CIFAR_STD),  
        transform.To(settings['dtype']),
        ]


    batch_size = config['batch_size']
    logging.info(f'Batch Size: {batch_size}')
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=False)
    process_trainset = preprocess(trainset, preprocess_train_transforms)
    process_testset = preprocess(testset, preprocess_test_transforms)
    logging.info(f"Training size= {len(trainset)}")
    trainloader = DataLoader(dataset=process_trainset, 
                            transforms=train_transforms, 
                            batch_size=batch_size, 
                            shuffle=True)       

    testloader = DataLoader(dataset=process_testset, 
                            batch_size=batch_size,
                            shuffle=False)


    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    criterion = config['criterion']
    criterion.cuda()
    model_name = config['model']
    # steps_per_epoch = int(steps_per_epoch * 1.0)
    logging.info(f"Model:{model_name}")
    if model_name =='skeleton':
        criterion = nn.CrossEntropyLoss(reduction='sum')
        criterion.cuda()
        model = get_skeleton_model(criterion)
        model.cuda().half()
    elif model_name == 'self-vanilla':
        model = ResNet(ResidualBlock, [1, 1, 1], initial_depth=64, batch_norm=config['batch_norm'])
        model.cuda().half()
    elif model_name == 'self-david':
        model = Network(batch_norm=config['batch_norm'], conv_bn=config['conv_bn'], activation=config['activation'])
        if config['half']:
            model.cuda().half()
        else:
            model.cuda()
    else:
        logging.error('incorrect model')
        sys.exit()
    
    logging.info("param size = %fMB", count_parameters_in_MB(model))
    
    # wandb.init(entity='wandb', project='resnet-dawnbench')
    # wandb.watch_called = False
    # wandb.watch(model, log="all", log_freq=3)
    # model.apply(weights_init_uniform)
    ret_dict = model_train(model, config, criterion, trainloader, testloader, testloader, model_name)

    # to_tensor = transforms.ToTensor()
    # x = to_tensor(trainset.data[0]).unsqueeze(0).cuda().to(dtype=torch.half)
    # with torch.autograd.profiler.profile(use_cuda=True) as profile:
    #     model(x)
    
    # logging.info(profile.key_averages().table())
    file_name = "experiments.txt"
    write_to_file(ret_dict, file_name)
    write_to_file(config, file_name)
    return ret_dict

if __name__ == '__main__':
    settings = get_dict()   
    config = dict()
    config['batch_size'] = settings['batch_size']
    config['budget'] = settings['budget']
    config['model'] = settings['name']
    config['swa'] = True
    config['swa_start'] = 25
    config['swa_step'] = 1
    config['weight_decay'] = 5e-5*config['batch_size'] #0# 
    config['momentum'] = 0.65
    config['swa_init_lr'] = 0.1
    config['base_lr'] = 0.065
    config['milestones'] = 'COSINE' #[0, config['budget']/5, config['budget']] #[0, int(config['swa_start']/2), config['swa_start'], 30] #[0, config['budget']/5, config['budget']] #[0, 5, config['budget']] #"COSINE" #[0, int(config['swa_start']/2), config['swa_start'], 30] #[0, 5, config['budget']]  #[0, int(config['swa_start']/2), config['swa_start'], 30][0, 5, config['budget']] #'cosine'
    config['schedule'] = 'COSINE' #[0, 0.1, 0] #[0, 0.2, config['swa_init_lr'], config['swa_init_lr']] #0, 0.2, 0] #[0, 0.1, 0] #"COSINE" #[0, 0.2, config['swa_init_lr'], config['swa_init_lr']] #[0, 1, 0] #[0, 0.2, config['swa_init_lr'], config['swa_init_lr']] #[0, 0.1, 0]
    config['batch_norm'] =  BatchNorm #partial(GhostBatchNorm, num_splits=16)
    config['activation'] = nn.ReLU  #partial(nn.CELU, alpha=0.075, inplace=False) # nn.ReLU  
    config['prefetch'] = True
    config['grad_clip'] = 5
    config['half'] = True
    config['conv_bn'] = conv_pool_bn_act #conv_bn_act_pool #conv_bn_pool_act #conv_pool_bn_act
    config['criterion'] =  nn.CrossEntropyLoss(reduction='sum')  #NMTCritierion(label_smoothing=0.2)#nn.CrossEntropyLoss(reduction='sum') #LabelSmoothLoss(smoothing=0.2)
    seeds = [1, 2, 42, 3]
    test_accuracies = list()
    train_accuracies = list()
    for seed in seeds:
        config['seed'] = seed
        return_dict = main(config)
        test_accuracies.append(return_dict['test_acc'])
        train_accuracies.append(return_dict['train_acc'])

    test_accuracies = np.array(test_accuracies)
    logging.info('TEST ACCURACY')
    logging.info('Mean of 4 runs', np.mean(test_accuracies)[0])
    logging.info('Std of 4 runs', np.std(test_accuracies)[0])
    train_accuracies = np.array(train_accuracies)
    logging.info('TRAIN ACCURACY')
    logging.info('Mean of 4 runs', np.mean(train_accuracies)[0])
    logging.info('Std of 4 runs', np.std(train_accuracies)[0])
