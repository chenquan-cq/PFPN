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
import math
import numpy as np
import warnings
import shutil
import traceback
import cv2
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import DataParallel
import random
import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torch.distributed as dist
import torch.utils.data
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import torchvision.transforms as tv_transforms

# working path
cur_path = osp.abspath(osp.dirname(__file__))
working_dir = osp.join(cur_path, '../')
sys.path.append(working_dir)

from utils import AverageMeter
from utils import vis_saliency_segment, dss_net_output_non_binary
from utils import joint_transforms, inverse_transform
from utils import extend_transforms
from utils import SaliencyEvaluation

from dataset import saliency
import networks


network_names = sorted(name for name in networks.__dict__
                       if not name.startswith("__")
                       and callable(networks.__dict__[name]))

# set mean and std value from ImageNet dataset
mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

# Testing settings
parser = argparse.ArgumentParser(description='Testing a Saliency network')

parser.add_argument('--max-size', dest='max_size',
                    help='max scale size (default: 256)', default=512, type=int)

parser.add_argument('--set-gpu', dest='set_gpu',
                    help='set gpu device (default: "0")', default='0', type=str)

parser.add_argument('--crop-size', dest='crop_size',
                    help='rand crop size (default: 256)', default=512, type=int)

parser.add_argument('--dataset-dir', dest='dataset_dir',
                    help='dataset dir (default: None)', default=None, type=str)

parser.add_argument('--datasets', nargs='+', default=[], required=True,
                    help='dataset names to merge')

parser.add_argument('--data-type', dest='data_type',
                    help='loading data by tfs, http or local', default='local', type=str)

parser.add_argument('-j', '--workers', dest='workers',
                    help='dataloader workers', default=4, type=int)

parser.add_argument('--input-normalize', dest='input_normalize', action='store_true',
                    help='normalizing input data')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--resume-dir', dest='resume_dir',
                    help='dir to latest checkpoint (default: None)', default=None, type=str)

parser.add_argument('--arch', '-a', metavar='ARCH', default=None, choices=network_names,
                    help='model architecture: ' + ' | '.join(network_names) + ' (default: None)')

parser.add_argument('--loss-func', dest='loss_func',
                    help='model loss function')

parser.add_argument('--side-output', dest='side_output', action='store_true')

parser.add_argument('--additional-output', dest='additional_output', action='store_true')

parser.add_argument('--vis', dest='vis', action='store_true',
                    help='visualize validation dataset')

parser.add_argument('--vis-dir', dest='vis_dir',
                    help='visualization dir (default: None)', default=None, type=str)

args = parser.parse_args()


def inference(func):
    def wrapper(*arg, **kwargs):
        args.cuda = not args.no_cuda and torch.cuda.is_available()

        global model
        model = networks.__dict__[args.arch](load_pretrained=False)
        model.loss = networks.__dict__[args.loss_func]()
        print('Create model "{}" done.'.format(args.arch))

        if args.cuda:
            model.cuda()
            model.loss.cuda()

        if args.resume_dir is not None:
            model_name_list = os.listdir(args.resume_dir)
            for model_name in model_name_list:
                if model_name_filter(model_name):
                    continue
                model_path = os.path.join(args.resume_dir, model_name)
                if os.path.isfile(model_path):
                    print("=> loading checkpoint '{}'".format(model_path))
                    checkpoint = torch.load(model_path)
                    try:
                        model.load_state_dict(checkpoint['state_dict'])
                    except Exception as e:
                        # print(e)
                        from collections import OrderedDict
                        mdict = OrderedDict()
                        for k, v in checkpoint['state_dict'].items():
                            assert (k.startswith('module.'))
                            nk = k[len('module.'):]
                            mdict[nk] = v
                        model.load_state_dict(mdict)

                    best_loss = checkpoint['best_loss']
                    print("=> loaded checkpoint '{}' (epoch {})"
                          .format(model_path, checkpoint['epoch']))
                else:
                    print('False checkpoint path...')
                    sys.exit(-1)
                func(*arg, **kwargs)
                print('####### {} ########'.format(model_name))
        else:
            if os.path.isfile(args.resume):
                print("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume)
                try:
                    model.load_state_dict(checkpoint['state_dict'])
                except Exception as e:
                    # print(e)
                    from collections import OrderedDict
                    mdict = OrderedDict()
                    for k, v in checkpoint['state_dict'].items():
                        assert (k.startswith('module.'))
                        nk = k[len('module.'):]
                        mdict[nk] = v
                    model.load_state_dict(mdict)

                best_loss = checkpoint['best_loss']
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(args.resume, checkpoint['epoch']))
            else:
                print('False checkpoint path...')
                sys.exit(-1)
            func(*arg, **kwargs)

    return wrapper


def model_name_filter(model_name):
    name_no_ext = model_name.split('.')[0]
    parts_list = name_no_ext.split('_')
    if parts_list[1] == 'epoch':
        if int(parts_list[2]) > 40:
            return False
        else:
            return True
    else:
        return True


@inference
def dataset_evaluation():
    global model
    saliency_evaluation = SaliencyEvaluation()

    dataset_names = args.datasets
    for dataset_name in dataset_names:
        saliency_evaluation.clear()
        save_path = os.path.join(args.vis_dir, dataset_name)
        if args.vis and not os.path.exists(save_path):
            print('Testing result path:{}'.format(save_path))
            os.makedirs(save_path)

        valid_jts = joint_transforms.Compose(
            [joint_transforms.Resize(args.crop_size)
             ])

        # valid source transforms
        valid_sts = tv_transforms.Compose(
            [
                extend_transforms.SingleResize(args.crop_size),
                extend_transforms.ImageToTensor(args.input_normalize),
                tv_transforms.Normalize(mean=mean, std=std)
            ])
        # valid target transforms
        valid_tts = extend_transforms.GrayImageToTensor()

        valid_dataset = saliency.SaliencyMergedData(
            root=args.dataset_dir,
            phase='valid',
            dataset_list=[dataset_name],
            joint_transform=None,
            source_transform=valid_sts,
            target_transform=valid_tts,
        )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            drop_last=True,
            batch_size=1,
            num_workers=args.workers,
            shuffle=False,
            pin_memory=True,
            sampler=None)

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # switch to evalutate mode
        model.eval()

        tbar = tqdm(valid_loader)
        for idx, data in enumerate(tbar):
            inputs, targets = data
            h, w = targets.size()[2:]
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            try:
                outputs = model(inputs)
            except Exception as e:
                continue
            # loss, main_loss = model.loss(outputs, targets)
            # losses.update(main_loss.data[0], inputs.size(0))

            # revise image scale
            if args.side_output:
                outputs = outputs[0]

            non_binary_output = dss_net_output_non_binary(outputs)[0]
            outputs = vis_saliency_segment(outputs, 197)[0]
            non_binary_output = cv2.resize(non_binary_output, (w, h))
            outputs = cv2.resize(outputs, (w, h))

            img = inverse_transform(inputs, mean, std)[0]
            anno = targets.detach().cpu().numpy()[0, 0, :, :] * 255
            # anno = anno.transpose(1, 2, 0)
            img = img[:, :, ::-1]
            img = cv2.resize(img, (w, h))

            # evaluation
            saliency_evaluation.add_one(non_binary_output, anno)

            if args.vis:
                img_name = valid_dataset.file_list[idx].split('\t')[0].split('/')[-1][:-4]
                # img_path = os.path.join(save_path, '{}.jpg'.format(img_name))
                # gt_path = os.path.join(save_path, '{}_gt.png'.format(img_name))
                mask_path = os.path.join(save_path, '{}_mask.jpg'.format(img_name))
                # pred_path = os.path.join(save_path, '{}_pred.jpg'.format(img_name))
                pred_path = os.path.join(save_path, '{}.png'.format(img_name))

                # cv2.imwrite(img_path, img)
                # cv2.imwrite(gt_path, anno)
                # cv2.imwrite(mask_path, img_mask)
                cv2.imwrite(pred_path, non_binary_output)

            tbar.set_description('Train loss: %.3f' % (losses.val))
            if idx % 1000 == 0:
                torch.cuda.empty_cache()

        MAE, Precision, Recall, F_m, S_m, E_m = saliency_evaluation.get_evaluation()

        idx = np.argmax(F_m)
        best_F = F_m[idx]
        mean_F = np.mean(F_m)
        best_precison = Precision[idx]
        best_recall = Recall[idx]
        print('{} - MAE:{}, max F-Measure:{}, mean F-Measure:{}, Precision:{},'
              ' Recall:{}, Threshold:{}, Loss:{},'
              ' S-Measure: {}, E-Measure: {}'
              .format(dataset_name, MAE, best_F, mean_F,
                      best_precison, best_recall, idx, losses.avg,
                      S_m, E_m))


if __name__ == '__main__':
    dataset_evaluation()
