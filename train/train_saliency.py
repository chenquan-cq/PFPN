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
import cv2
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import DataParallel
import random
import time
import torch.backends.cudnn as cudnn
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
from utils import joint_transforms
from utils import extend_transforms
from utils import localtime

from networks.scheduler import GradualWarmupScheduler, IsometryScheduler, ExponentialScheduler
from dataset import saliency
from dataset import sampler
import networks

network_names = sorted(name for name in networks.__dict__
                       if not name.startswith("__")
                       and callable(networks.__dict__[name]))

__all__ = [
        'get_parser',
        'Trainer',
        ]

def get_parser():
    # Training settings
    parser = argparse.ArgumentParser(description='Train a Deep Saliency Detection Model')

    # data
    parser.add_argument('--dataset-dir', dest='dataset_dir',
                        help='dataset dir (default: None)', default=None, type=str)
    parser.add_argument('--train-label-dir', dest='train_label_dir',
                        help='train label dir (default: None)', default=None, type=str)
    parser.add_argument('--valid-label-dir', dest='valid_label_dir',
                        help='valid label dir (default: None)', default=None, type=str)
    parser.add_argument('--input-normalize', dest='input_normalize', action='store_true',
                        help='normalizing input data')
    parser.add_argument('--data-type', dest='data_type',
                        help='loading data by tfs, http or local', default='local', type=str)
    parser.add_argument('--crop-size', dest='crop_size',
                        help='rand crop size (default: 256)', default=256, type=int)
    parser.add_argument('--max-size', dest='max_size',
                        help='max scale size (default: 256)', default=256, type=int)
    parser.add_argument('--ignore-label', dest='ignore_label',
                        help='ignore label(default: 255)', default=255, type=int)
    parser.add_argument('--weighted-sampler', dest='weighted_sampler', action='store_true',
                        help='using weighted sampler')
    parser.add_argument('--datasets_weight', nargs='+', default=[],
                        help='dataset weights for weighted sampler')
    parser.add_argument('--train-num-sampler', dest='train_num_sampler',
                        help='total wighted sampler number for train (default: None)', default=None, type=int)
    parser.add_argument('--valid-num-sampler', dest='valid_num_sampler',
                        help='total wighted sampler number for valid (default: None)', default=None, type=int)
    parser.add_argument('-j', '--workers', dest='workers',
                        help='dataloader workers', default=4, type=int)
    parser.add_argument('--epochs', dest='epochs',
                        help='number of epochs to train (default: 100)', default=100, type=int)
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', dest='batch_size',
                        help='mini-batch size for training (default: 8)', default=8, type=int)
    parser.add_argument('--test-batch-size', dest='test_batch_size',
                        help='batch size for test (default: 1)', default=1, type=int)
    parser.add_argument('--train-conf', dest='train_conf',
                        help='train conf filename (default: None)', default=None, type=str)
    parser.add_argument('--valid-conf', dest='valid_conf',
                        help='valid conf filename (default: None)', default=None, type=str)
    parser.add_argument('--datasets', nargs='+', default=[], required=True,
                        help='dataset names to merge')

    # optimizer
    parser.add_argument('--optimizer-strategy', dest='optimizer_strategy',
                        help='optimizer strategy (default: SGD)', default='SGD', type=str)
    parser.add_argument('--using-multi-lr', dest='using_multi_lr', action='store_true',
                        help='using multi-lr and multi-weight-decay strategy defined in model')
    parser.add_argument('--lr', '--learning-rate', dest='lr',
                        default=1e-6, type=float,
                        help='initial learning rate (default: 1e-2/sqrt(2))')
    parser.add_argument('--resume-lr-decay', dest='resume_lr_decay', action='store_true',
                        help='learning rate decay when restoring from ckpt')
    parser.add_argument('--lr-decay', dest='lr_decay',
                        help='learning rate decay(default: 0.9)', default=0.9, type=float)
    parser.add_argument('--last-lr', dest='last_lr',
                        help='last learning rate (default: 0.001/sqrt(2))', default=0.001 / (math.sqrt(16. / 8)),
                        type=float)
    parser.add_argument('--last-lr-decay', dest='last_lr_decay',
                        help='last learning rate decay(default: 0.9)', default=0.9, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='SGD momentum (default: 0.9)', default=0.9, type=float)
    parser.add_argument('--wd', '--weight-decay', dest='weight_decay',
                        help='weight decay (default: 0.0001)', default=1e-4, type=float)
    parser.add_argument('--last-weight-decay', dest='last_weight_decay',
                        help='last weight decay (default: 0.0001)', default=0.0005, type=float)
    parser.add_argument('--reset-optimizer', dest='reset_optimizer', action='store_true',
                        help='reset optimizer from resume checkpoint')
    parser.add_argument('--reset-lr', dest='reset_lr', action='store_true',
                        help='reset learning rate of optimizer from resume checkpoint')
    parser.add_argument('--lr-scheduler', dest='lr_scheduler', action='store_true',
                        help='adopt learning rate decay strategy')
    parser.add_argument('--lr-scheduler-iter', dest='lr_scheduler_iter', action='store_true',
                        help='adopt learning rate decay strategy per iteration')
    parser.add_argument('--lr-decay-steps', dest='lr_decay_steps', type=int, nargs='+',
                        default=[], help='epoch milestones for learning rate decay')

    # model and loss
    parser.add_argument('--arch', '-a', metavar='ARCH', default=None, choices=network_names,
                        help='model architecture: ' + ' | '.join(network_names) + ' (default: None)')
    parser.add_argument('--loss-func', '-loss_func',
                        help='model loss function')
    parser.add_argument('--out-chn', dest='num_out_chn',
                        help='the number of output channels (default: 2)', default=2, type=int)
    parser.add_argument('--save-dir', dest='save_dir',
                        help='save dir (default: None)', default=None, type=str)
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained-model', dest='pretrained_model',
                        help='pretrained model path (default: None)', default=None, type=str)
    parser.add_argument('--feed-label', dest='feed_label', action='store_true')

    # other
    parser.add_argument('--vis', dest='vis', action='store_true',
                        help='visualize validation dataset')
    parser.add_argument('--vis-dir', dest='vis_dir',
                        help='visualization dir (default: None)', default=None, type=str)
    parser.add_argument('--vis-loss', dest='vis_loss', action='store_true',
                        help='visualize loss in tensorboard')
    parser.add_argument('--root-dir', dest='root_dir',
                        help='root dir for T-Net(default: None)', default=None, type=str)
    parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=None, metavar='S',
                        help='random seed for initializing training (default: None)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('-dist', '--distributed', dest='distributed', action='store_true',
                        help='distributed training')
    parser.add_argument('--multi-gpu', dest='multi_gpu', action='store_true',
                        help='multi gpu training')
    parser.add_argument('--user', dest='user',
                        help='user name, like chenquan.cq', default=None, type=str)
    parser.add_argument('--set-gpu', dest='set_gpu',
                        help='set gpu device (default: "0")', default='0', type=str)

    return parser


class Trainer():
    def __init__(self, parser):
        self.parser = parser
        self.args = self.parser.parse_args()
        self.best_loss = 1e8
        self.best_epoch = 0
        self.args.cuda = not self.args.no_cuda and torch.cuda.is_available()

        if self.args.seed is not None:
            random.seed(self.args.seed)
            np.random.seed(self.args.seed)
            torch.manual_seed(self.args.seed)
            if self.args.cuda:
                torch.cuda.manual_seed(self.args.seed)
                cudnn.deterministic = True
            warnings.warn('You have chosen to seed training. '
                          'This will turn on the CUDNN deterministic setting, '
                          'which can slow down your training considerably! '
                          'You may see unexpected behavior when restarting '
                          'from checkpoints.')

        # set trained model save dir
        if self.args.save_dir is None:
            print('Save dir must be given')
            self.parser.print_help()
            sys.exit(-1)
        if self.args.arch is None:
            print('Model arch must be given')
            self.parser.print_help()
            sys.exit(-1)

        # add sweeping parameters into save dir
        self.dir_appendix = '{}_{}ep_{}lr'.format(
            self.args.optimizer_strategy, self.args.epochs, self.args.lr)
        self.args.save_dir = os.path.join(self.args.save_dir, self.dir_appendix)
        if not os.path.exists(self.args.save_dir):
            print('args.save_dir:{}'.format(self.args.save_dir))
            os.makedirs(self.args.save_dir)
        print('Run by args: \n\t{}'.format(self.args))

        if self.args.vis_loss:
            self.writer = SummaryWriter(self.args.log_dir)
        else:
            self.writer = None

        # build model
        self.build()

        # load dataset
        self.load_dataset()

        # init optimizer
        self.init_optimizer()

        # init scheduler
        self.init_lr_scheduler()

        if self.args.resume:
            self.load_ckpt()

    def build(self):
        self.model = networks.__dict__[self.args.arch]()
        self.model_loss = networks.__dict__[self.args.loss_func](
            weight=[1.0,0.5,0.3,0.3,0.3,0.3],
        )
        print('Create model "{}" done.'.format(self.args.arch))
        if self.args.cuda:
            self.model.cuda()
            self.model_loss.cuda()

            if self.args.multi_gpu:
                self.model = torch.nn.parallel.DataParallel(self.model)
                self.model_loss = torch.nn.parallel.DataParallel(self.model_loss)

    def load_dataset(self):
        # set mean and std value from ImageNet dataset
        if self.args.input_normalize:
            rgb_mean, rgb_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            rgb_mean, rgb_std = [122.675, 116.669, 104.008], [58.395, 57.12, 57.375]

        # train joint transforms
        train_jts = joint_transforms.Compose(
            [  # joint_transforms.ElasticTransform(),
                joint_transforms.Resize(self.args.max_size),
                joint_transforms.Apply(tv_transforms.Lambda(
                    lambda img: img.astype(np.float) / 255),
                    th=[False, True]),
                joint_transforms.RandomCrop(self.args.crop_size, ignore_label=255),
                joint_transforms.RandomHorizontallyFlip(),
            ])
        # train source transforms
        train_sts = tv_transforms.Compose(
            [  # extend_transforms.RandomGaussianBlur(blur_prob=0.1),
                # extend_transforms.RandomBright(),
                extend_transforms.ImageToTensor(self.args.input_normalize),
                tv_transforms.Normalize(mean=rgb_mean, std=rgb_std)
            ])
        # train target transforms
        # train_tts = extend_transforms.MapToTensor()
        train_tts = extend_transforms.GrayImageToTensor(False)

        train_dataset = saliency.SaliencyMergedData(
            root=self.args.dataset_dir,
            phase='train',
            dataset_list=self.args.datasets,
            joint_transform=train_jts,
            source_transform=train_sts,
            target_transform=train_tts,
        )

        self.train_sampler = None
        if self.args.weighted_sampler:
            self.train_sampler = sampler.DatasetWeightedSampler(
                train_dataset.weight_list, self.args.train_num_sampler)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            drop_last=True,
            num_workers=self.args.workers,
            shuffle=(self.train_sampler is None),
            pin_memory=True,
            sampler=self.train_sampler,
        )

        # valid joint transforms
        valid_jts = joint_transforms.Compose(
            [joint_transforms.Resize(self.args.crop_size),
             # joint_transforms.RandomCrop(args.crop_size),
             ])
        # valid source transforms
        valid_sts = tv_transforms.Compose(
            [extend_transforms.ImageToTensor(self.args.input_normalize),
             tv_transforms.Normalize(mean=rgb_mean, std=rgb_std)
             ])
        # valid target transforms
        valid_tts = extend_transforms.GrayImageToTensor()

        valid_dataset = saliency.SaliencyMergedData(
            root=self.args.dataset_dir,
            phase='valid',
            dataset_list=self.args.datasets,
            joint_transform=valid_jts,
            source_transform=valid_sts,
            target_transform=valid_tts,
        )

        self.valid_sampler = None
        if self.args.weighted_sampler:
            self.valid_sampler = sampler.DatasetWeightedSampler(
                valid_dataset.weight_list, self.args.valid_num_sampler)

        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            drop_last=True,
            batch_size=self.args.test_batch_size,
            num_workers=self.args.workers,
            shuffle=False,
            pin_memory=True,
            sampler=self.valid_sampler,
        )

    def init_optimizer(self):
        # optimizer
        if self.args.using_multi_lr:
            train_params = self.model.lr_list(self.args.lr) \
                if not self.args.distributed and not self.args.multi_gpu \
                else self.model.module.lr_list(self.args.lr)
        else:
            train_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.args.optimizer_strategy == 'SGD':
            self.optimizer = optim.SGD(
                train_params, self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=0.9, nesterov=True)
        elif self.args.optimizer_strategy == 'Adam':
            self.optimizer = optim.Adam(
                train_params, self.args.lr,
                weight_decay=self.args.weight_decay)
        else:
            self.optimizer = optim.Adadelta(
                train_params, self.args.lr,
                weight_decay=self.args.weight_decay)

    def init_lr_scheduler(self):
        # iteration
        max_iter = len(self.train_loader) * self.args.epochs

        # lr scheduler
        if self.args.lr_scheduler and not self.args.lr_scheduler_iter:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.args.lr_decay_steps)
        elif self.args.lr_scheduler and self.args.lr_scheduler_iter:
            self.scheduler = ExponentialScheduler(self.optimizer, max_iter, exponent=0.9)
        else:
            self.scheduler = None

    def load_ckpt(self):
        print("=> loading checkpoint '{}'".format(self.args.resume))
        checkpoint = torch.load(self.args.resume)
        try:
            self.model.load_state_dict(checkpoint['state_dict'])
        except Exception as e:
            print(e)
            from collections import OrderedDict
            mdict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                assert (k.startswith('module.'))
                nk = k[len('module.'):]
                mdict[nk] = v
                self.model.load_state_dict(mdict)

        self.best_loss = checkpoint['best_loss']
        if self.args.reset_optimizer:
            self.args.start_epoch = checkpoint['epoch']
            self.best_epoch = checkpoint['best_epoch']
            print('reset optimizer, only model and best loss are resumed.')
        else:
            self.args.start_epoch = checkpoint['epoch']
            self.best_epoch = checkpoint['best_epoch']
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler is not None and 'scheduler' in checkpoint.keys():
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if self.args.resume_lr_decay:
                for idx, g in enumerate(self.optimizer.param_groups):
                    self.optimizer.param_groups[idx]['lr'] *= 0.1
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(self.args.resume, checkpoint['epoch']))

    def run(self):
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # adjust learning rate
            if self.args.lr_scheduler and not self.args.lr_scheduler_iter:
                self.scheduler.step()

            # train for one epoch
            train_loss = self.train_validate(epoch, is_train=True)
            # evaluate on validation set
            # valid_loss = 1e8
            if self.valid_loader is not None:
                valid_loss = self.train_validate(epoch, is_train=False)
            else:
                valid_loss = 1.5

            # remember the best loss and save checkpoint
            is_best = False
            if valid_loss < self.best_loss:
                is_best = True
                self.best_loss = valid_loss
                self.best_epoch = epoch
            filename = os.path.join(
                self.args.save_dir,
                '{}_epoch_{}_train_loss_{:.4f}_valid_loss_{:.4f}.pth.tar'
                    .format(self.args.arch, epoch, train_loss, valid_loss))
            print('Current epoch {} valid loss:{}, best epoch {} valid loss:{}'.format(
                epoch, valid_loss, self.best_epoch, self.best_loss))
            if self.scheduler is None:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_loss': self.best_loss,
                    'best_epoch': self.best_epoch,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, filename, train_loss)
            else:
                self.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.state_dict(),
                    'best_loss': self.best_loss,
                    'best_epoch': self.best_epoch,
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                }, is_best, filename, train_loss)

    def train_validate(self, epoch, is_train=True):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        # switch to train mode
        if is_train:
            self.model.train()
            curr_iter = epoch * len(self.train_loader)
            data_loader = self.train_loader
        else:
            self.model.eval()
            curr_iter = epoch * len(self.valid_loader)
            data_loader = self.train_loader

        end = time.time()
        dataset_length = len(data_loader)
        for bid, data in enumerate(data_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            inputs, targets = data
            if self.args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)

            # compute output
            outputs = self.model(inputs)
            loss, main_loss = self.model_loss(outputs, targets)

            # compute gradient and do SGD step
            if is_train:
                self.optimizer.zero_grad()
                if self.args.multi_gpu:
                    record_loss = main_loss.data.mean()
                    loss.mean().backward()
                else:
                    record_loss = main_loss.data
                    loss.backward()
                # record loss
                losses.update(record_loss, inputs.size(0))
                # torch.nn.utils.clip_grad_norm(model.parameters(), args.clip_gradient)
                self.optimizer.step()
            else:
                if self.args.multi_gpu:
                    record_loss = main_loss.data.mean()
                else:
                    record_loss = main_loss.data
                # record loss
                losses.update(record_loss, inputs.size(0))


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if is_train:
                log_prefix = '{0}\tEpoch: [{1}] [{2}/{3}]'.format(
                    localtime(), epoch, bid, dataset_length
                )
            else:
                log_prefix = '{0}\tValid: [{1}/{2}]'.format(
                    localtime(), bid, dataset_length)
            if bid % self.args.log_interval == 0 or bid == (dataset_length-1):
                print(
                    '{}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss_val:.4f} ({loss_avg:.4f})\t'.format(
                        log_prefix,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss_val=losses.val,
                        loss_avg=losses.avg,
                    )
                )
            if self.args.vis_loss and is_train:
                self.writer.add_scalar('train', losses.val, curr_iter)
            elif self.args.vis_loss and not is_train:
                self.writer.add_scalar('eval', losses.val, curr_iter)
            if is_train and self.args.lr_scheduler_iter:
                self.scheduler.step()
            curr_iter += 1

        return losses.avg

    def save_checkpoint(self, state, is_best, filename, train_loss):
        best_path = os.path.join(
            self.args.save_dir,
            '{}_best_occur_epoch_{}_train_loss_{}_valid_loss_{}.pth.tar'
                .format(self.args.arch, state['best_epoch'], train_loss, state['best_loss']))

        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, best_path)

if __name__ == '__main__':
    trainer = Trainer(get_parser())
    trainer.run()
