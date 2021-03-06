import torch
import numpy as np
from torch.optim import lr_scheduler 
import math
import matplotlib.pyplot as plt

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
        

class SWAResNetLR(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, schedule, swa_start, base_lr, swa_init_lr, swa_step, eta_min=0, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        if len(milestones) != len(schedule):
            raise AssertionError('Length of milestones should be equal to the length of schedule')
        self.eta_min = eta_min
        self.milestones = milestones
        self.schedule = schedule
        self.swa_start, self.swa_init_lr, self.swa_step= swa_start, swa_init_lr, swa_step
        self.base_lr = base_lr
         
        super(SWAResNetLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        
        if self.last_epoch >= self.milestones[-1]:
            return [self.base_lr for base_lr in self.base_lrs]
        
        if self.last_epoch > self.swa_start: 
            return np.interp([self.last_epoch%self.swa_step], [0, self.swa_step], [self.swa_init_lr, self.base_lr])
        else:
            return np.interp([self.last_epoch], self.milestones, self.schedule)       