import logging
import datetime
from torch import nn
import torchvision
logging.basicConfig(level=logging.INFO)
import torch
from utils import AverageMeter, accuracy, plot_classes_preds
from dataset import DataPrefetchLoader
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
from settings import get

# import sys

def train(trainloader, model, criterion, optimizer, name, clip, prefetch=True):
    if prefetch:
        trainloader = DataPrefetchLoader(trainloader)
        top1_avg, objs_avg, a = train_prefetch(trainloader, model, criterion, optimizer, clip)
    elif name == 'skeleton':
        top1_avg, objs_avg, a = train_skeleton(trainloader, model, optimizer, clip)
    elif 'self' in name:
        top1_avg, objs_avg, a = train_self(trainloader, model, criterion, optimizer,clip)
        
    return top1_avg, objs_avg, a

def train_prefetch(trainloader, model, criterion, optimizer, clip):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    inputs, targets = trainloader.next()
    # print(f'shape:{inputs.shape}')
    step = 0
    while inputs is not None:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        # if step % report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)            
        inputs, targets = trainloader.next()
    
    a = datetime.datetime.now() - c
    return top1.avg, objs.avg, a


def train_self(trainloader, model, criterion, optimizer, clip):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    for step, (inputs, target) in enumerate(trainloader, 0):
        # zero the parameter gradients
        optimizer.zero_grad()

        device = get('device')
        dtype = get('dtype')
        if device.type != 'cpu':
            target = target.cuda(non_blocking=True)
        
        inputs = inputs.to(dtype=dtype, device=device)
        logits = model(inputs)

        loss = criterion(logits, target)
        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        # if step % report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)  
    a = datetime.datetime.now() - c
    return top1.avg, objs.avg, a

def train_skeleton(trainloader, model, optimizer, clip):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    for step, (inputs, target) in enumerate(trainloader, 0):

        # zero the parameter gradients
        optimizer.zero_grad()
        device = get('device')
        if device.type != 'cpu':
            target = target.cuda(non_blocking=True)
        
        inputs = inputs.to(dtype=torch.half, device=device)
        logits, loss = model(inputs, target)

        loss.sum().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = inputs.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        # if step % report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)            
    a = datetime.datetime.now() - c
    return top1.avg, objs.avg, a

def infer(valid_queue, model, criterion, name, prefetch=True):
    if prefetch:
        valid_queue = DataPrefetchLoader(valid_queue)
        top1_avg, objs_avg, a = infer_prefetch(valid_queue, model, criterion)
    elif name == 'skeleton':
        top1_avg, objs_avg, a = infer_skeleton(valid_queue, model)
    elif 'self' in name:
        top1_avg, objs_avg, a = infer_self(valid_queue, model, criterion)
    
    return top1_avg, objs_avg, a

def infer_prefetch(valid_queue, model, criterion):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()
    inputs, targets = valid_queue.next()
    step = 0
    with torch.no_grad():
        while inputs is not None:
            logits = model(inputs)
            loss = criterion(logits, targets)

            prec1, prec5 = accuracy(logits, targets, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            # if step % report_freq == 0:
            #     logging.info(f'{type}, {step}, {objs.avg}, {top1.avg}, {top5.avg}')
            inputs, targets = valid_queue.next()
    a = datetime.datetime.now() - c            
    return top1.avg, objs.avg, a

def infer_self(valid_queue, model, criterion):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():    
        for step, (inputs, target) in enumerate(valid_queue):
            
            dtype = get('dtype')
            device = get('device')
            if device.type != 'cpu':
                target = target.cuda(non_blocking=True)
            
            inputs = inputs.to(dtype=dtype, device=device)

            logits = model(inputs)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            # if step % report_freq == 0:
            #     logging.info(f'{type}, {step}, {objs.avg}, {top1.avg}, {top5.avg}')
    a = datetime.datetime.now() - c            
    return top1.avg, objs.avg, a

def infer_skeleton(valid_queue, model):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():    
        for step, (inputs, target) in enumerate(valid_queue):
            device = get('device')
            if device.type != 'cpu':
                target = target.cuda(non_blocking=True)
            
            inputs = inputs.to(dtype=torch.half, device=device)

            logits, loss = model(inputs, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = inputs.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

    a = datetime.datetime.now() - c            
    return top1.avg, objs.avg, a
