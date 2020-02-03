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

import numpy as np
import torch
from torch import nn
import time


__all__ = [
        'initialize_weights',
        'vis_saliency_segment',
        'dss_net_output_non_binary',
        'print_network',
        'denormalize_to_images',
        'inverse_transform',
        'localtime',
        ]

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                # nn.init.kaiming_normal_(module.weight, mode='fan_in',
                #         nonlinearity='relu')
                nn.init.kaiming_normal(module.weight, mode='fan_in')
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

def denormalize_to_images(tensor, mean, std):
    """visualize a B x C x H x W normlized image tensor where C=3 to an image"""
    image_list = []
    for i in range(tensor.size(0)):
        ts = torch.squeeze(tensor[i, :, :, :])
        for t, m, s in zip(ts, mean, std):
            t.mul_(s).add_(m)
        image = ts.cpu().numpy() * 255
        image = np.transpose(image, [1, 2, 0]).astype(np.uint8)
        image = image[:, :, ::-1]
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_list.append(image)
    return image_list

def dss_net_output_non_binary(tensor):
    res_list = []
    for i in range(tensor.size(0)):
        seg_compact = torch.squeeze(tensor[i, :, :, :].cpu().data).numpy()
        seg_compact *= 255
        res_list.append(seg_compact)
    return res_list

def vis_saliency_segment(tensor, threshold=0):
    res_list = []
    for i in range(tensor.size(0)):
        seg_compact = torch.squeeze(tensor[i, :, :, :].cpu().data).numpy()
        seg_compact *= 255
        seg = np.zeros(seg_compact.shape, dtype=np.uint8)
        seg[np.where(seg_compact > threshold)] = 255
        res_list.append(seg)
    return res_list


def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of parameters: {}".format(num_params))

def inverse_transform(img_tensor, mean, std):
    """
    transform img_tensor to unscaled img
    input:
        img_tensor - Num x Channel x Height x Width, tensor
        mean - Channel, list
        std - Channel, list
    output:
        img_list - a list including Num items,
                   each is a Height x Width x Channel array(rgb, np.uint8)
    """
    assert len(img_tensor.shape) == 4
    assert img_tensor.shape[1] == len(mean)
    assert img_tensor.shape[1] == len(std)

    for chn_id, (m_val, s_val) in enumerate(zip(mean, std)):
        img_tensor[:,chn_id,:,:] = img_tensor[:,chn_id,:,:]*s_val + m_val

    img_tensor = torch.clamp(img_tensor, 0, 1)
    img = img_tensor.cpu().numpy().transpose([0,2,3,1])*255
    return list(img.astype(np.uint8))


def localtime():
    """local time: yyyy-mm-dd hh:mm:ss"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
