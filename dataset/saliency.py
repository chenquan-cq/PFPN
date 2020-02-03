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
import os
import sys
import numpy as np
import cv2
import os.path as osp
import urllib.request
import requests as req
from io import BytesIO
import torch
from torch.utils import data
import contextlib
import traceback

def ParamCandidateCheck(param, cands):
    if param not in cands:
        raise RuntimeError('Candiate param: {}'.format(cands))


class SaliencyMergedData(data.Dataset):
    def __init__(self, root, phase,
                 dataset_list,
                 joint_transform=None,
                 source_transform=None,
                 target_transform=None):
        ParamCandidateCheck(phase, ['train', 'valid', 'test'])
        if not isinstance(dataset_list, list):
            raise RuntimeError('dataset_list must be a list')

        dataset_names = []
        dataset_list_iter = iter(dataset_list)
        try:
            for item in dataset_list_iter:
                dataset_names.append(item)

        except Exception as e:
            print('Dataset list error:{}'.format(e))

        self.root = root
        self.joint_transform = joint_transform
        self.source_transform = source_transform
        self.target_transform = target_transform

        self.file_list = []
        self.weight_list = []
        for idx in range(len(dataset_names)):
            name = dataset_names[idx]

            print(root, name, phase)
            conf_path = os.path.join(root, name + '_' + phase + '.conf')
            # print('conf_path:{}'.format(conf_path))

            with open(conf_path) as fin:
                curr_list = []
                try:
                    for line in fin:
                        curr_list.append(line.strip())
                except Exception as e:
                    print(traceback.format_exc())
                    print(e)
                # curr_list = fin.read().strip().split('\n')
                print('Dataset {} has {} samples.'.format(name, len(curr_list)))
                self.file_list.extend(curr_list)

        print('\nDataset SaliencyMergedData has {} samples.'.format(
            len(self.file_list)))
        return

    def __getitem__(self, index):
        item = self.file_list[index].split('\t')

        # binaray = int(item[2])

        img_path = os.path.join(self.root, item[0])
        mask_path = os.path.join(self.root, item[1])
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print('image is None: {}'.format(img_path))
        if mask is None:
            print('mask is None: {}'.format(mask_path))
        # if len(mask.shape) == 2:
        #     mask = np.expand_dims(mask, 2)

        # bgr to rgb
        img = img[:, :, ::-1]
        # joint transforms -> source transforms -> target transforms
        """
        Attention:
            make annotation before transformation
            because of the existence of ignore_label
        """
        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)
        if self.source_transform is not None:
            img = self.source_transform(img)
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return img, mask

    def __len__(self):
        return len(self.file_list)
