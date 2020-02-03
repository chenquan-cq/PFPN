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

import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import os
import numpy as np
import random
sys.path.append(os.path.dirname(__file__))

__all__ = ['SaliencyLoss',
        ]

class SaliencyLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=255):
        super(SaliencyLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, output, label):
        if isinstance(output, list):
            if self.weight is not None:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = self.weight[0] * F.binary_cross_entropy(output[0][label!=self.ignore_index],
                                                               label[label!=self.ignore_index])
                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += self.weight[i] * F.binary_cross_entropy(x[label!=self.ignore_index],
                                                                    label[label!=self.ignore_index])
            else:
                label = F.upsample(label, size=output[0].size()[2:])
                loss = F.binary_cross_entropy(output[0][label!=self.ignore_index],
                                              label[label!=self.ignore_index])
                main_loss = loss.clone()
                for i, x in enumerate(output[1:]):
                    label = F.upsample(label, size=x.size()[2:])
                    loss += F.binary_cross_entropy(x[label!=self.ignore_index],
                                                   label[label!=self.ignore_index])
            return loss, main_loss
        else:
            label = F.upsample(label, size=output.size()[2:])
            loss = F.binary_cross_entropy(output[label!=self.ignore_index],
                                          label[label!=self.ignore_index])
            main_loss = loss
        return loss, main_loss
