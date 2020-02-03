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
import argparse
import os
import os.path as osp
import sys
import numpy as np
import traceback
from collections import OrderedDict
import cv2

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as tv_transforms

from utils import dss_net_output_non_binary
from utils import extend_transforms
from utils import print_network
import networks

network_names = sorted(name for name in networks.__dict__
                       if not name.startswith("__")
                       and callable(networks.__dict__[name]))

# set mean and std value from ImageNet dataset
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

class PFPN_Inference(object):
    def __init__(self, model_path, use_cuda=True):
        super(PFPN_Inference, self).__init__()
        self.use_cuda = use_cuda
        self.model = networks.__dict__['PFPN'](load_pretrained=False)
        print('Create model "{}" done.'.format('PFPN'))

        if use_cuda:
            self.model.cuda()

        print("=> loading checkpoint '{}'".format(model_path))
        checkpoint = torch.load(model_path)
        try:
            new_dict = OrderedDict()
            for k, _ in self.model.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    new_dict[k] = checkpoint['state_dict'][k]
            self.model.load_state_dict(new_dict)
        except Exception as e:
            new_dict = OrderedDict()
            for k, _ in self.model.state_dict().items():
                if 'num_batches_tracked' in k:
                    new_dict[k] = torch.zeros(1)
                else:
                    nk = 'module.' + k
                    new_dict[k] = checkpoint['state_dict'][nk]
            self.model.load_state_dict(new_dict)

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(model_path, checkpoint['epoch']))

        self.image_tf = tv_transforms.Compose(
            [extend_transforms.SingleResize(256),
             extend_transforms.ImageToTensor(),
             tv_transforms.Normalize(mean=mean, std=std)
             ])

        self.model.eval()

    def predict(self, img):
        '''
        :param img: rgb image
        :return: mask 0,255
        '''
        h, w = img.shape[:2]

        # bgr to rgb
        inputs = img[:, :, ::-1]
        inputs = self.image_tf(inputs)
        inputs = inputs.unsqueeze(0)
        if self.use_cuda:
            inputs = inputs.cuda()
        inputs = Variable(inputs)

        # compute output
        with torch.no_grad():
            outputs = self.model(inputs)

        outputs = outputs[0]
        non_binary_output = dss_net_output_non_binary(outputs)[0]
        non_binary_output = cv2.resize(non_binary_output, (w, h))

        return non_binary_output

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Testing a Saliency network')

    parser.add_argument('--use-cuda', dest='use_cuda', action='store_true',
                        help='use CUDA')

    parser.add_argument('--ckpt-path', dest='ckpt_path', default=None, type=str,
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--image-path', dest='image_path',
                        help='image path for predicting', default=None, type=str)

    parser.add_argument('--vis-dir', dest='vis_dir',
                        help='visualization directory of predicted result (default: None)',
                        default='.', type=str)

    args = parser.parse_args()

    if args.ckpt_path is None or not os.path.exists(args.ckpt_path):
        os.makedirs('models/pretrained', exist_ok=True)
        os.system('wget #url -P ./models/pretrained/PRNet_epoch_59_train_loss_0.0214_valid_loss_0.1717.pth.tar')
        args.ckpt_path = 'models/pretrained/PRNet_epoch_59_train_loss_0.0214_valid_loss_0.1717.pth.tar'
    if args.image_path is None:
        args.image_path = 'ILSVRC2012_test_00097358.jpg'

    model = PFPN_Inference(args.ckpt_path, use_cuda=args.use_cuda)
    image = cv2.imread(args.image_path)
    saliency_map = model.predict(image)
    save_path = os.path.join(
        args.vis_dir,
        os.path.splitext(args.image_path)[0]+'_pred.png')
    cv2.imwrite(save_path, saliency_map)

