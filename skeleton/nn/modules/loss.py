# -*- coding: utf-8 -*-
# pylint: disable=arguments-differ, abstract-method
from __future__ import absolute_import
import logging

import torch


LOGGER = logging.getLogger(__name__)


class CrossEntropyLabelSmooth(torch.nn.Module):
    def __init__(self, num_classes, epsilon, reduction='avg'):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, input, target):  # pylint: disable=redefined-builtin
        log_probs = self.logsoftmax(input)
        targets = torch.zeros_like(log_probs).scatter_(1, target.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs)

        if self.reduction == 'avg':
            loss = loss.mean(0).sum()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
