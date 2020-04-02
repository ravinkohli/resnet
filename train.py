import logging
import datetime
from torch import nn
logging.basicConfig(level=logging.INFO)
import torch
from utils import AverageMeter, accuracy, plot_classes_preds
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train(trainloader, model, criterion, optimizer, report_freq=50):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.train()
    for step, (input, target) in enumerate(trainloader, 0):

        # zero the parameter gradients
        optimizer.zero_grad()
        input = input.cuda()
        target = target.cuda(non_blocking=True)
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

def infer(valid_queue, model, criterion, type='valid', report_freq=50):
    c = datetime.datetime.now()
    objs = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    with torch.no_grad():    
        for step, (input, target) in enumerate(valid_queue):
            input = input.cuda()
            target = target.cuda(non_blocking=True)
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
