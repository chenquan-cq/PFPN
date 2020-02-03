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
import torchvision
import numpy as np
import cv2
import sys
import os
from collections import OrderedDict

from networks.layers import BatchNormFixed, _normal_layer
from networks.config import *

__all__ = [
           'ResNet',
           'ResNet101',
        ]

class ResNet(nn.Module):
    def __init__(self, use_bn=True, fix_bn=False,
                 strides=[1, 2, 2, 2],
                 layers=[3, 4, 6, 3],
                 dilations=[1, 1, 1, 1],
                 pretrained=None):
        super(ResNet, self).__init__()
        self.expansion = 4
        self.inplanes = 64
        self.use_bn = use_bn
        self.fix_bn = fix_bn
        if self.fix_bn:
            self.bn_type = 'fixed_bn'
        else:
            self.bn_type = 'normal'
        self.pretrained = pretrained
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2),
                               padding=(3, 3), bias=False)
        self.bn1 = BatchNormFixed(64) if self.fix_bn else nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0], stride=strides[0], dilation=dilations[0])
        self.layer2 = self._make_layer(128, layers[1], stride=strides[1], dilation=dilations[1])
        self.layer3 = self._make_layer(256, layers[2], stride=strides[2], dilation=dilations[2])
        self.layer4 = self._make_layer(512, layers[3], stride=strides[3], dilation=dilations[3])
        # self.avgpool = nn.AvgPool2d(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        num_params = 0
        for p in self.parameters():
            num_params += p.numel()
        print("The number of parameters in ResNet: {}".format(num_params))
        if self.pretrained is not None:
            self.load_pretrained()

    def forward(self, x):
        encoder = []
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        encoder.append(x)
        x = self.layer1(x)
        encoder.append(x)
        x = self.layer2(x)
        encoder.append(x)
        x = self.layer3(x)
        encoder.append(x)
        x = self.layer4(x)
        encoder.append(x)

        return encoder

    def _make_layer(self, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * self.expansion:
            if self.use_bn:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, dilation=dilation,
                              stride=stride, bias=False),
                    BatchNormFixed(planes * self.expansion) if self.fix_bn else \
                        nn.BatchNorm2d(planes * self.expansion),
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * self.expansion,
                              kernel_size=1, stride=stride,
                              dilation=dilation, bias=False),
                )

        layers = []
        layers.append(BottleNeck(self.inplanes, planes, stride, dilation, downsample,
                                 use_bn=self.use_bn, bn_type=self.bn_type))
        self.inplanes = planes * self.expansion
        for i in range(1, blocks):
            layers.append(BottleNeck(self.inplanes, planes,
                                     use_bn=self.use_bn, bn_type=self.bn_type))

        return nn.Sequential(*layers)

    def load_pretrained(self):
        if not os.path.exists(self.pretrained):
            raise RuntimeError('Please ensure {} exists.'.format(
                self.pretrained))
        checkpoint = torch.load(self.pretrained)

        try:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    new_dict[k] = checkpoint[k]
            self.load_state_dict(new_dict)
        except Exception as e:
            new_dict = OrderedDict()
            for k, _ in self.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    nk = k[len('module'):]
                    new_dict[k] = checkpoint[nk]
            self.load_state_dict(new_dict)
        print("=> loaded checkpoint '{}'".format(self.pretrained))

class ResNet101(ResNet):
    def __init__(self, use_bn=True, fix_bn=False,
                 strides=[1, 2, 2, 2],
                 dilations=[1, 1, 1, 1],
                 pretrained=None):
        super(ResNet101, self).__init__(
            use_bn=use_bn, fix_bn=fix_bn,
            strides=strides,
            layers=[3, 4, 23, 3],
            dilations=dilations,
            pretrained=pretrained)

    def load_pretrained(self):
        if not os.path.exists(self.pretrained):
            print('download pretrained resent101 to {}'.format(
                os.path.basename(self.pretrained)
            ))
            os.system(
                'wget https://download.pytorch.org/models/resnet101-5d3b4d8f.pth'
                ' -P {}'.format(os.path.dirname(self.pretrained)))
        super(ResNet101, self).load_pretrained()

class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, use_bn=True, momentum=0.1,
                 bn_type='normal'):
        super(BottleNeck, self).__init__()
        self.use_bn = use_bn
        self.bn_type = bn_type
        self.momentum = momentum
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1,
                               dilation=dilation, bias=False)
        if self.use_bn:
            self.bn1 = self.normal_layer(planes, momentum=self.momentum)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.relu2 = nn.ReLU(inplace=True)
        if self.use_bn:
            self.bn2 = self.normal_layer(planes, momentum=self.momentum)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1,
                               dilation=dilation, bias=False)
        if self.use_bn:
            self.bn3 = self.normal_layer(planes*4, momentum=self.momentum)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        if self.use_bn:
            out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        if self.use_bn:
            out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

    def normal_layer(self, *args, **kwargs):
        return _normal_layer(self.bn_type, *args, **kwargs)

