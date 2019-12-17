import torch
from torch import nn

class FixedAffine(torch.nn.Module):
    def __init__(self,factor,offset):
        torch.nn.Module.__init__(self)
        self.register_buffer('factor',torch.tensor(factor))
        self.register_buffer('offset',torch.tensor(offset))

    def forward(self, x):
        return x*self.factor+self.offset

    def __repr__(self):
        return 'FixedAffine(factor={:f},offset={:f})'.format(self.factor,self.offset)