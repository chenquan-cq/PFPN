#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) 2019 Alibaba Group Holding Limited.
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
##############################################################################

from __future__ import print_function
import torch
import torch.nn as nn
from torch.nn import init
import numpy as np

__all__ = [
        '_normal_layer',
        'BatchNormFixed',
        'layer_weights_init',
        ]

def _normal_layer(bn_type, *args, **kwargs):
    if bn_type == 'fixed_bn':
        return BatchNormFixed(*args, **kwargs)
    elif bn_type == 'group':
        return nn.GroupNorm(*args, **kwargs)
    else:
        return nn.BatchNorm2d(*args, **kwargs)

class BatchNormFixed(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 use_global_status=True):
        super(BatchNormFixed, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.use_global_status = use_global_status
        if self.affine:
            self.weight = nn.Parameter(torch.Tensor(num_features))
            self.bias = nn.Parameter(torch.Tensor(num_features))
            self.weight.data.uniform_()
            self.bias.data.zero_()
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, input):
        if self.affine:
            if self.training and not self.use_global_status:
                mean = input.mean(dim=0, keepdim=True)\
                    .mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
                var = (input-mean)*(input-mean).mean(dim=0, keepdim=True).\
                    mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)

                self.running_mean = (mean.squeeze()
                                     + (1-self.momentum)*self.running_mean)/(2-self.momentum)
                batch = input.size()[0]
                bias_correction_factor = batch-1 if batch>1 else 1
                self.running_var = (var.squeeze() * bias_correction_factor
                                    + (1-self.momentum)*self.running_var)/(2-self.momentum)
                x = input - mean
                x = x / (var+self.eps).sqrt()

                tmp_weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                tmp_bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                x = x * tmp_weight
                x = x + tmp_bias
            else:
                tmp_running_mean = self.running_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                tmp_running_var = self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                x = input - tmp_running_mean
                x = x / (tmp_running_var+self.eps).sqrt()

                tmp_weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                tmp_bias = self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                x = x * tmp_weight
                x = x + tmp_bias
        else:
            x = input
        return x

    def extra_repr(self):
        s = ('{num_features}, eps={eps}, momentum={momentum}'
             ', affine={affine}, use_global_status={use_global_status}')
        return s.format(**self.__dict__)


def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype='float32')
    # weight[range(in_channels), range(out_channels), :, :] = filt
    for i in range(in_channels):
        for j in range(out_channels):
            weight[i, j, :, :] = filt
    return torch.from_numpy(weight)

def layer_weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        # init.normal(m.weight.data, mean=0, std=0.01)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1)
        if m.bias is not None:
            init.constant_(m.bias, 0)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                length = len(param)
                nn.init.constant(param, 0.0)
                nn.init.constant(param[length // 4:length // 2], 1.0)
            elif 'weight' in name:
                nn.init.uniform(param, -0.2, 0.2)
                # nn.init.xavier_normal(param)
    elif isinstance(m, nn.ConvTranspose2d):
        size = m.weight.data.size()
        m.weight.data = bilinear_kernel(size[0], size[1], size[2])
        if m.bias is not None:
            init.constant_(m.bias, 0)



if __name__ == '__main__':
    layer = BatchNormFixed(3, use_global_status=False)
    layer.train()

    # print(layer.running_mean.size())
    input = torch.ones(1, 3, 3, 3)
    output = layer(input)
    print(output, layer.running_mean.size())
