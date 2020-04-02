import torch
import numpy as np
from torch.optim import lr_scheduler 
import math
import matplotlib.pyplot as plt
from model import ResNet, ResidualBlock

class PiecewiseLinearLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones=[0, 15, 30, 35], schedule=[0, 0.1, 0.005, 0], eta_min=0, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        if len(milestones) != len(schedule):
            raise AssertionError('Length of milestones should be equal to the length of schedule')
        self.eta_min = eta_min
        self.milestones = milestones
        self.schedule = schedule
        super(PiecewiseLinearLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        
        if self.last_epoch >= self.milestones[-1]:
            return [self.eta_min for base_lr in self.base_lrs]
        
        return np.interp([self.last_epoch], self.milestones, self.schedule)
        # i = [self.last_epoch[i] >= self.milestones[i] for i in range(len(self.milestones))]
        # return (self.schedule[i+1]-self.schedule[i])/(self.milestones[i+1]- self.milestones[i]) * self.last_epoch
        

# def get_change_scale(scheduler, init_scale=1.0):
#     def schedule(e, scale=None, **kwargs):
#         lr = scheduler(e, **kwargs)
#         return lr * (scale if scale is not None else init_scale)
#     return schedule


# def get_piecewise(knots, vals):
#     def schedule(e, **kwargs):
#         return np.interp([e], knots, vals)[0]
#     return schedule