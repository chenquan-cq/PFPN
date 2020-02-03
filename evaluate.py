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

import os
import numpy as np
import cv2
import argparse
from utils import evaluation


parser = argparse.ArgumentParser(description='Testing a Saliency network')

parser.add_argument('--data-dir', dest='data_dir', default=None, type=str,
                    help='data root directory')

parser.add_argument('--res-dir', dest='res_dir', default=None, type=str,
                    help='predicted result directory')

parser.add_argument('--datasets', nargs='+', default=[], required=True,
                    help='dataset names to merge')

# parser.add_argument('--paths-file', dest='paths_file',
#                     help='file name in root directory, '
#                          'include predicted saliency map paths and groundtruth paths',
#                     default=None, type=str)

args = parser.parse_args()

if args.data_dir is None:
    raise RuntimeError('data-dir can not be empty')
if args.res_dir is None:
    raise RuntimeError('res-dir can not be empty')

data_dir = args.data_dir
res_dir = args.res_dir

for dataset in args.datasets:
    saliency_evaluation = evaluation.SaliencyEvaluation()
    saliency_evaluation.clear()
    with open(os.path.join(args.data_dir, dataset+'_valid.conf')) as f:
        for line in f:
            line = line.strip().split('\t')
            gt_path = line[1]
            pred_path = os.path.join(dataset, gt_path.split('/')[-1])

            pred = cv2.imread(os.path.join(res_dir, pred_path), cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(os.path.join(data_dir, gt_path), cv2.IMREAD_GRAYSCALE)
            saliency_evaluation.add_one(
                    pred.astype(np.float), gt.astype(np.float))
    MAE, Precision, Recall, F_m, S_m, E_m = saliency_evaluation.get_evaluation()

    idx = np.argmax(F_m)
    best_F = F_m[idx]
    mean_F = np.mean(F_m)
    best_precison = Precision[idx]
    best_recall = Recall[idx]
    print('{} - MAE:{}, max F-Measure:{}, mean F-Measure:{}, Precision:{},'
          ' Recall:{}, S-Measure: {}, E-Measure: {}'
          .format(dataset, MAE, best_F, mean_F, best_precison, best_recall,
                  S_m, E_m))


