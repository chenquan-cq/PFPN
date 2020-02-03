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

from networks.layers import layer_weights_init, BatchNormFixed
from networks.backbone import ResNet, ResNet101
from networks.config import *

__all__ = ['PFPN',
        ]

Normal_Type = 'batch'
Momentum = 0.1
def _normal_layer(in_ch):
    if Normal_Type == 'fixed_bn':
        return BatchNormFixed(in_ch)
    elif Normal_Type == 'group':
        return nn.GroupNorm(in_ch, in_ch)
    else:
        return nn.BatchNorm2d(in_ch, momentum=Momentum)


class PFPN(nn.Module):
    def __init__(self, load_pretrained=True):
        super(PFPN, self).__init__()
        if load_pretrained:
            self.pretrained = model_addr['res101']
        else:
            self.pretrained = None
        self.base = PFPNBase(fix_bn=True,
                           strides=[1, 2, 2, 2],
                           dilations=[1, 1, 1, 1],
                           pretrained=self.pretrained)
        branch_planes = [256, 256, 256, 256, 256]
        self.transition = Transition(
            inplanes=[64, 256, 512, 1024, 2048],
            outplanes=branch_planes)
        self.fpm1 = FeaturePolishingModule(
            inplanes=branch_planes,
            outplanes=branch_planes,
            size_times=[1, 2, 4, 8, 16])
        self.fpm2 = FeaturePolishingModule(
            inplanes=branch_planes,
            outplanes=branch_planes,
            size_times=[1, 2, 4, 8, 16])
        self.transition2 = Transition(
            inplanes=branch_planes,
            outplanes=[32]*len(branch_planes),
            is_scale=True
        )

        merge_planes = 32 * len(branch_planes)
        self.conv = nn.Sequential(
            nn.Conv2d(merge_planes, merge_planes, 3, padding=1),
            _normal_layer(merge_planes),
            # nn.ReLU(inplace=True),
            nn.Conv2d(merge_planes, merge_planes, 3, padding=1),
            _normal_layer(merge_planes),
            # nn.ReLU(inplace=True)
        )
        self.output_layer = OutputLayer(
            inplanes=[merge_planes]+branch_planes)

        if load_pretrained:
            self.init_weight()

    def forward(self, input):
        encoders = self.base(input)
        outputs, _ = self.transition(encoders)
        outputs = self.fpm1(outputs)
        outputs = self.fpm2(outputs)

        fuse_output, outputs = self.transition2(outputs, input.size()[2:])
        fuse_output = torch.cat(fuse_output, 1)
        fuse_output = self.conv(fuse_output)
        outputs = [fuse_output] + outputs

        outputs = self.output_layer(outputs)
        # outputs = outputs[::-1]

        return outputs

    def init_weight(self):
        self.transition.apply(layer_weights_init)
        self.fpm1.apply(layer_weights_init)
        self.fpm2.apply(layer_weights_init)
        self.transition2.apply(layer_weights_init)
        self.conv.apply(layer_weights_init)
        self.output_layer.apply(layer_weights_init)

    def lr_list(self, lr=1.0):
        lr_list = []
        lr_list.append({'params': self.base.parameters(), 'lr': 0.1 * lr})
        lr_list.append({'params': self.transition.parameters()})
        lr_list.append({'params': self.fpm1.parameters()})
        lr_list.append({'params': self.fpm2.parameters()})
        lr_list.append({'params': self.transition2.parameters()})
        lr_list.append({'params': self.conv.parameters()})
        lr_list.append({'params': self.output_layer.parameters()})

        return lr_list


class PFPNBase(ResNet101):
    def forward(self, x):
        encoder = []

        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = self.relu(x)
        encoder.append(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        encoder.append(x)
        x = self.layer2(x)
        encoder.append(x)
        x = self.layer3(x)
        encoder.append(x)
        x = self.layer4(x)
        encoder.append(x)

        return encoder


class FeaturePolishingModule(nn.Module):
    def __init__(self, inplanes=[], outplanes=[], size_times=[],
                 mode='up', multi_branch=True):
        super(FeaturePolishingModule, self).__init__()
        self.branches = len(inplanes)
        self.mode = mode
        self.multi_branch = multi_branch
        self.branch_merges = []
        if self.mode == 'down':
            if not multi_branch:
                self.branch_merges.append(
                    BranchConcat(
                        inplanes, outplanes[-1],
                        size_times=size_times,
                        mode=self.mode))
            else:
                for i in range(self.branches):
                    self.branch_merges.append(
                        BranchConcat(
                            inplanes[:i+1], outplanes[i],
                            size_times=size_times[:i+1],
                            mode=self.mode))
        else:
            if not multi_branch:
                self.branch_merges.append(
                    BranchConcat(
                        inplanes, outplanes[0],
                        size_times=size_times,
                        mode=self.mode))
            else:
                for i in range(self.branches):
                    self.branch_merges.append(
                        BranchConcat(
                            inplanes[i:], outplanes[i],
                            size_times=size_times[i:],
                            mode=self.mode))
        self.branch_merges = nn.ModuleList(self.branch_merges)

    def forward(self, inputs):
        outputs = []
        if self.multi_branch:
            if self.mode == 'down':
                for i in range(self.branches):
                    outputs.append(self.branch_merges[i](inputs[:i+1]))
            else:
                for i in range(self.branches):
                    outputs.append(self.branch_merges[i](inputs[i:]))
        else:
            outputs.append(self.branch_merges[0](inputs))
        return outputs


class BranchConcat(nn.Module):
    def __init__(self, inplanes=[], outplanes=256, size_times=[],
                 mode='down'):
        super(BranchConcat, self).__init__()
        self.in_branches = len(inplanes)
        self.mode = mode
        if mode == 'down':
            is_downsample = [item < size_times[-1] for item in size_times]
            is_upsample = [item > size_times[-1] for item in size_times]
        else:
            is_downsample = [item < size_times[0] for item in size_times]
            is_upsample = [item > size_times[0] for item in size_times]
        self.conv_list = []
        for i, ch in enumerate(inplanes):
            if is_downsample[i]:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(ch, outplanes,
                                  kernel_size=3,
                                  padding=1),
                        _normal_layer(outplanes),
                        nn.ReLU(),
                        DownsampleModule(outplanes, outplanes, size_times[i],
                                         kernel_size=3, stride=2, padding=1)
                    ))
            elif is_upsample[i]:
                self.conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(ch, outplanes, kernel_size=3, padding=1),
                        _normal_layer(outplanes),
                        nn.ReLU(),))
                        # UpsampleModule(outplanes, int(size_times[i] / size_times[0])),))
            else:
                self.conv_list.append(
                    nn.Sequential(nn.Conv2d(ch, outplanes,
                                            kernel_size=3,
                                            padding=1),
                                  _normal_layer(outplanes),
                                  nn.ReLU()))
        self.conv_list = nn.ModuleList(self.conv_list)
        self.final_conv = nn.Conv2d(outplanes*len(inplanes), outplanes,
                                    kernel_size=1)
        self.final_bn = _normal_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        if self.mode == 'down':
            size = inputs[-1].size()[2:]
        else:
            size = inputs[0].size()[2:]

        output = []
        for input, branch_conv in zip(inputs, self.conv_list):
            output.append(F.upsample(branch_conv(input), size,
                          mode='bilinear', align_corners=True))
        output = torch.cat(output, 1)
        output = self.final_bn(self.final_conv(output))
        if self.mode == 'down':
            output += inputs[-1]
        else:
            output += inputs[0]
        output = self.relu(output)
        return output


class Transition(nn.Module):
    def __init__(self, inplanes=[], outplanes=[],
                 is_scale=False):
        super(Transition, self).__init__()
        assert len(inplanes) == len(outplanes)
        self.is_scale = is_scale
        self.conv_list = []
        for in_ch, out_ch in zip(inplanes, outplanes):
            self.conv_list.append(
                nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=1),
                              _normal_layer(out_ch))
            )
        self.conv_list = nn.ModuleList(self.conv_list)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input, size=None):
        if self.is_scale:
            for idx in range(len(input)):
                input[idx] = F.upsample(input[idx], size,
                                        mode='bilinear')
        upsample = input
        output = []
        for idx, x in enumerate(input):
            output.append(self.conv_list[idx](x))

        return output, upsample


class UpsampleModule(nn.Module):
    def __init__(self, in_ch, size, upsample_type='upsample'):
        super(UpsampleModule, self).__init__()
        self.upsample_type = upsample_type
        self.size = size
        if self.upsample_type == 'ConvTranspose':
            self.dec_conv = nn.ConvTranspose2d(in_ch, in_ch, 4, stride=2,
                                               padding=1, bias=False, groups=in_ch)
        elif self.upsample_type == 'upsampleConv':
            self.dec_conv = nn.Conv2d(in_ch, in_ch,
                                      kernel_size=3,
                                      padding=1, bias=False)

    def forward(self, input):
        size = input.size()[2:]
        size = [item * self.size for item in size]
        if self.upsample_type == 'ConvTranspose':
            output = self.dec_conv(input)
        elif self.upsample_type == 'upsampleConv':
            output = F.upsample(input, size, mode='bilinear', align_corners=True)
            output = self.dec_conv(output)
        else:
            output = F.upsample(input, size, mode='bilinear', align_corners=True)

        return output


class DownsampleModule(nn.Module):
    def __init__(self, inplanes, outplanes,
                 size_times, kernel_size, dilation=1,
                 stride=1, padding=0, bias=True):
        super(DownsampleModule, self).__init__()
        self.conv = nn.Conv2d(inplanes, outplanes,
                              kernel_size=kernel_size,
                              dilation=dilation,
                              stride=stride,
                              padding=padding,
                              bias=bias)
        self.bn = _normal_layer(outplanes)

    def forward(self, input):
        output = F.relu(self.bn(self.conv(input)))
        return output


class OutputLayer(nn.Module):
    def __init__(self, inplanes=[]):
        super(OutputLayer, self).__init__()
        self.conv_list = []
        for planes in inplanes:
            self.conv_list.append(nn.Conv2d(planes, 1, 1))
        self.conv_list = nn.ModuleList(self.conv_list)

    def forward(self, inputs):
        outputs = []
        for i, input in enumerate(inputs):
            outputs.append(F.sigmoid(self.conv_list[i](input)))
        return outputs


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PFPN()
    # print([name for name, param in model.named_parameters()])
    model = model.to(device)
    input = torch.randn(1, 3, 128, 128)
    pred = model(input)
    print(pred[0].size())
