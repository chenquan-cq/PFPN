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
import traceback

__all__ = [
        'AverageMeter',
        'SaliencyEvaluation',
        ]

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SaliencyEvaluation(object):
    def __init__(self):
        self.max_threshold = 255
        self.epsilon = 1e-8
        self.beta = 0.3
        self.sm = SM()
        self.em = EM()

        self.MAE = 0
        self.Precision = np.zeros(256)
        self.Recall = np.zeros(256)
        self.S_Measure = 0
        self.E_Measure = 0
        self.num = 0.0

    def add_one(self, predict, gt):
        try:
            MAE, Precision, Recall, _, S_Measure, E_Measure \
                = self.evaluation(predict, gt)
            self.MAE += MAE
            self.Precision += Precision
            self.Recall += Recall
            self.S_Measure += S_Measure
            self.E_Measure += E_Measure
            self.num += 1.0
        except:
            print(traceback.print_exc())

    def clear(self):
        self.MAE = 0
        self.Precision = np.zeros(256)
        self.Recall = np.zeros(256)
        self.S_Measure = 0
        self.E_Measure = 0
        self.num = 0

    def get_evaluation(self):
        if self.num > 0:
            avg_MAE = self.MAE / self.num
            avg_Precision = self.Precision / self.num
            avg_Recall = self.Recall / self.num
            avg_S_Measure = self.S_Measure / self.num
            avg_E_Measure = self.E_Measure / self.num
            F_m = (1.3 * avg_Precision * avg_Recall) / (0.3 * avg_Precision + avg_Recall)
            return avg_MAE, avg_Precision, avg_Recall,\
                   F_m, avg_S_Measure, avg_E_Measure
        else:
            return 0, np.zeros(256), np.zeros(256), np.zeros(256), 0, 0

    def evaluation(self, predict, gt):
        MAE = self.mae(predict, gt)
        # gt[gt > 25] = 255
        # gt[gt <= 25] = 0
        Precision, Recall = self.precision_and_recall(predict, gt)
        S_Measure = self.sm.eval(predict, gt)
        E_Measure = self.em.eval(predict, gt)
        return MAE, Precision, Recall, 0, S_Measure, E_Measure

    def mae(self, predict, gt):
        '''
            predict: numpy, shape is (height, width), value 0-255
            gt: numpy, shape is (height, width), value 0-255
        '''
        assert predict.shape == gt.shape
        return np.mean(np.abs(predict - gt)/255.0)

    def f_measure(self, precision, recall):
        f = ((1 + self.beta) * precision * recall) / (self.beta * precision + recall)
        return f

    def precision_and_recall(self, predict, gt, threshold=None):
        '''
            predict: numpy, shape is (height, width), value 0-255
            gt: numpy, shape is (height, width), value 0-255
        '''
        assert predict.shape == gt.shape
        pred_max = np.max(predict)
        if pred_max > 0:
            predict[predict < 0] = 0
            predict = predict.astype(np.float)
            predict *= 255.0/pred_max
            predict = np.round(predict).astype(np.int)

        if threshold is None:
            if pred_max > 0:
                GT = np.zeros(gt.shape, dtype=np.int)
                GT[gt > 0] = 1
                predict_th = np.cumsum(np.bincount(
                    predict.flatten(), minlength=256)[::-1])
                predict_precision_th = np.cumsum(np.bincount(
                    predict[GT==1].flatten(), minlength=256)[::-1])
                precision = predict_precision_th / predict_th.astype(np.float)
                recall = predict_precision_th/ float(np.sum(GT))
                return precision[::-1], recall[::-1]
            else:
                return np.zeros(256), np.zeros(256)

        else:
            # ground true pixel
            TS = np.zeros(predict.shape)
            # predicted true pixel
            DS = np.zeros(predict.shape)

            TS[gt > (self.max_threshold / 2)] = 1
            DS[predict > threshold] = 1
            TSDS = TS * DS

            precision = (np.mean(TSDS)+self.epsilon) / (np.mean(DS)+self.epsilon)
            recall = (np.mean(TSDS)+self.epsilon) / (np.mean(TS)+self.epsilon)
            return precision, recall

class SM(object):
    def __init__(self):
        self.alpha = 0.5
        self.eps = 1e-8

    def eval(self, predict, gt):
        dGT = gt.astype(np.float) / 255.0
        y = dGT.mean()

        if y == 0:
            x = (predict / 255.0).mean()
            Q = 1.0 - x
        elif y == 1:
            x = (predict / 255.0).mean()
            Q = x
        else:
            alpha = 0.5
            Q = alpha * self.so(predict, gt) \
                + (1 - alpha) * self.sr(predict, gt)
            if Q < 0:
                Q = 0

        return Q

    def so(self, predict, gt):
        prediction_fg = predict.copy() / 255.0
        prediction_fg[gt == 0] = 0
        O_FG = self.object(prediction_fg, gt)

        prediction_bg = 1.0 - predict / 255.0
        prediction_bg[gt > 0] = 0
        O_BG = self.object(prediction_bg, gt == 0)

        u = (gt / 255.0).mean()
        Q = u * O_FG + (1 - u) * O_BG

        return Q

    def object(self, predict, gt):
        x = (predict[gt > 0]).mean()
        sigma_x = (predict[gt > 0]).std()

        score = 2.0 * x / (x**2 + 1.0 + sigma_x + self.eps)

        return score

    def sr(self, predict, gt):
        X, Y = self.centroid(gt)

        GT_1, GT_2, GT_3, GT_4, w1, w2, w3, w4 \
            = self.divideGT(gt, X, Y)

        prediction_1, prediction_2, prediction_3, prediction_4 \
            = self.Divideprediction(predict, X, Y)

        Q1 = self.ssim(prediction_1, GT_1)
        Q2 = self.ssim(prediction_2, GT_2)
        Q3 = self.ssim(prediction_3, GT_3)
        Q4 = self.ssim(prediction_4, GT_4)

        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        return Q

    def ssim(self, predict, gt):
        hei, wid = predict.shape[:2]
        N = wid * hei
        prediction = predict.copy()
        dGT = gt.astype(np.float)

        x = (prediction).mean()
        y = (dGT).mean()

        # sigma_x2 = var(prediction(:))
        sigma_x2 = ((prediction - x)**2).sum() / (N - 1 + self.eps)
        # sigma_y2 = var(dGT(:))
        sigma_y2 = ((dGT - y)**2).sum() / (N - 1 + self.eps)

        sigma_xy = ((prediction - x) * (dGT - y)).sum() / (N - 1 + self.eps)

        alpha = 4 * x * y * sigma_xy
        beta = (x**2 + y**2) * (sigma_x2 + sigma_y2)

        if alpha != 0:
            Q = alpha / (beta + self.eps)
        elif alpha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

    def Divideprediction(self, predict, X, Y):
        hei, wid = predict.shape[:2]

        LT = predict[:Y, :X].copy() / 255.0
        RT = predict[:Y, X:wid].copy() / 255.0
        LB = predict[Y:hei, :X].copy() / 255.0
        RB = predict[Y:hei, X:wid].copy() / 255.0

        return LT, RT, LB, RB

    def divideGT(self, gt, X, Y):
        hei, wid = gt.shape[:2]
        area = wid * hei

        LT = gt[:Y, :X].copy() / 255.0
        RT = gt[:Y, X: wid].copy() / 255.0
        LB = gt[Y:hei, :X].copy() / 255.0
        RB = gt[Y:hei, X:wid].copy() / 255.0

        w1 = (X * Y) / area
        w2 = ((wid - X) * Y) / area
        w3 = (X * (hei - Y)) / area
        w4 = 1.0 - w1 - w2 - w3

        return LT, RT, LB, RB, w1, w2, w3, w4

    def centroid(self, gt):
        rows, cols = gt.shape[:2]
        dGT = gt.astype(np.float) / 255.0

        if gt.sum() == 0:
            X = round(cols / 2)
            Y = round(rows / 2)
        else:
            total = (gt.astype(np.float) / 255.0).sum()
            i = np.array(range(cols))
            j = np.array(range(rows))
            X = int(round((dGT.sum(0) * i).sum() / total))
            Y = int(round((dGT.sum(1) * j).sum() / total))

        return X, Y

class EM(object):
    def __init__(self):
        self.eps = 1e-8

    def eval(self, predict, gt):
        '''
        @conference{Fan2018Enhanced, title={Enhanced-alignment Measure for Binary Foreground Map Evaluation},
                    author={Fan, Deng-Ping and Gong, Cheng and Cao, Yang and Ren, Bo and Cheng, Ming-Ming and Borji, Ali},
                    year = {2018},
                    booktitle = {IJCAI}
        }
        Input:
            predict - Binary/Non binary foreground map with values in the range [0 255]. Type: double.
            gt - Binary/Non binary foreground map with values in the range [0 255]. Type: double, uint8.
        Output:
            score - The score
        '''
        dGT = gt.astype(np.bool).astype(np.float)
        th = 2 * predict.mean()
        dFG = np.zeros_like(predict)
        dFG[predict > th] = 1
        dFG = dFG.astype(np.bool).astype(np.float)

        if dGT.sum() == 0:
            enhanced_matrix = 1.0 - dFG
        elif (dGT == 0).sum() == 0:
            enhanced_matrix = dFG
        else:
            align_matrix = self.AlignmentTerm(dFG, dGT)
            enhanced_matrix = self.EnhancedAlignmentTerm(align_matrix)

        h, w = gt.shape[:2]
        score = enhanced_matrix.sum()/ (w * h - 1 + self.eps)

        return score

    def AlignmentTerm(self, dFM, dGT):
        mu_FM = dFM.mean()
        mu_GT = dGT.mean()

        align_FM = dFM - mu_FM
        align_GT = dGT - mu_GT

        align_Matrix = 2. * (align_GT * align_FM) / (align_GT * align_GT + align_FM * align_FM + self.eps)
        return align_Matrix

    def EnhancedAlignmentTerm(self, align_Matrix):
        enhanced = ((align_Matrix + 1)**2) / 4
        return enhanced
