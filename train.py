import logging
from torch import nn
logging.basicConfig(level=logging.INFO)
import torch
from utils import AverageMeter, accuracy, plot_classes_preds
torch.set_default_tensor_type('torch.cuda.FloatTensor')

def train(trainloader, model, criterion, optimizer, writer, report_freq=50):
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

        if step % report_freq == 0:
            logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
            # ...log the running loss
            writer.add_scalar('training loss',
                            objs.avg,
                            step)

            # ...log a Matplotlib Figure showing the model's predictions on a
            # random mini-batch
            writer.add_figure('predictions vs. actuals',
                            plot_classes_preds(model, input, target),
                            global_step=step)
        
    return top1.avg, objs.avg

def infer(valid_queue, model, criterion, report_freq=50):
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

            if step % report_freq == 0:
                logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg
