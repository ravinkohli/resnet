import logging
import datetime
from torch import nn
import torchvision
logging.basicConfig(level=logging.INFO)
import torch
from utils import AverageMeter, accuracy, plot_classes_preds
torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.autograd.set_detect_anomaly(True)
# import sys

def train(trainloader, model, criterion, optimizer, name):
    if name == 'skeleton':
        top1_avg, objs_avg, a = train_skeleton(trainloader, model, optimizer)
    else:
        top1_avg, objs_avg, a = train_self(trainloader, model, criterion, optimizer)
    return top1_avg, objs_avg, a

def train_self(trainloader, model, criterion, optimizer):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    for step, (input, target) in enumerate(trainloader, 0):

        # zero the parameter gradients
        # print(input.shape)
        optimizer.zero_grad()
        # torchvision.utils.save_image(input, f'input_{step}.png')
        # sys.exit()
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        # input.to(dtype=torch.half)
        logits = model(input)

        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        # if step % report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)            
    a = datetime.datetime.now() - c
    return top1.avg, objs.avg, a

def train_skeleton(trainloader, model, optimizer):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    for step, (input, target) in enumerate(trainloader, 0):

        # zero the parameter gradients
        optimizer.zero_grad()
        input = input.cuda().to(dtype=torch.half)
        target = target.cuda(non_blocking=True)
        # input.to(dtype=torch.half)
        logits, loss = model(input, target)

        loss.sum().backward()
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        # if step % report_freq == 0:
        #     logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)            
    a = datetime.datetime.now() - c
    return top1.avg, objs.avg, a

def infer(valid_queue, model, criterion, name, type='valid'):
    if name == 'skeleton':
        top1_avg, objs_avg, a = infer_skeleton(valid_queue, model, type='valid')
    else:
        top1_avg, objs_avg, a = infer_self(valid_queue, model, criterion, type='valid')
    return top1_avg, objs_avg, a

def infer_self(valid_queue, model, criterion, type='valid'):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():    
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
            # # if name == 'skeleton':
            # input.to(dtype=torch.half)
            logits = model(input)
            loss = criterion(logits, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            # if step % report_freq == 0:
            #     logging.info(f'{type}, {step}, {objs.avg}, {top1.avg}, {top5.avg}')
    a = datetime.datetime.now() - c            
    return top1.avg, objs.avg, a

def infer_skeleton(valid_queue, model, type='valid'):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():    
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda().to(dtype=torch.half)
            target = target.cuda(non_blocking=True)

            logits, loss = model(input, target)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

    a = datetime.datetime.now() - c            
    return top1.avg, objs.avg, a
