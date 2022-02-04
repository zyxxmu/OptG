from torch.nn import init
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

from args import args as parser_args
import numpy as np

DenseConv = nn.Conv2d

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, prune_rate):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()#
        j = int(prune_rate * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class SparseConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #mask_real = self.weight.data.new(self.weight.size())
        #self.mask = nn.Parameter(mask_real)
        self.mask = nn.Parameter(torch.zeros(self.weight.shape))
        self.prune_rate = 0

    def forward(self, x):
        subnet = GetSubnet.apply(self.mask, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def getSparsity(self, f=torch.sigmoid):
        subnet = GetSubnet.apply(self.mask, self.prune_rate)
        w = self.weight * subnet
        temp = w.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel()

    def set_prune_rate(self, prune_rate):
        self.prune_rate = prune_rate

class GetMySubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, prune_index, preserve_index):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[prune_index] = 0
        flat_out[preserve_index] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None

class MySparseConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #mask_real = self.weight.data.new(self.weight.size())
        #self.mask = nn.Parameter(mask_real)
        self.mask = nn.Parameter(torch.zeros(self.weight.shape))
        self.prune_index = []
        self.preserve_index = [] 
        self.prune_rate = 0

    def forward(self, x):
        subnet = GetMySubnet.apply(self.mask, self.prune_index, self.preserve_index)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

    def set_prune_rate(self, prune_rate):
        # Get the subnetwork by sorting the scores and using the top k%
        scores = self.mask
        out = scores.clone()
        _, idx = scores.flatten().sort()#descending=True 从大到小排序，descending=False 从小到大排序，默认 descending=Flase
        j = int(prune_rate * scores.numel())
        self.prune_index = idx[:j]
        self.preserve_index = idx[j:]

    def getSparsity(self, f=torch.sigmoid):
        subnet = GetMySubnet.apply(self.mask, self.prune_index, self.preserve_index)
        w = self.weight * subnet
        temp = w.detach().cpu()
        temp[temp!=0] = 1
        return (100 - temp.mean().item()*100), temp.numel()

