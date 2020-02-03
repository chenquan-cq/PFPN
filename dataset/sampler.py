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

import math
import heapq
import random
import torch
from torch.utils.data.sampler import Sampler
from torch.distributed import get_world_size, get_rank

class WeightedDistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    Arguments:
        dataset: Dataset used for sampling.
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """
    def __init__(self, dataset, weights, total_num_samples, num_replicas=None, rank=None):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        if total_num_samples is not None and total_num_samples < len(self.dataset):
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        else:
            self.num_samples = int(math.ceil(len(total_num_samples) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

        self.weights = torch.tensor(weights, dtype=torch.double)

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.multinomial(self.weights,
                                    self.num_samples * self.num_replicas,
                                    replacement=False)
        indices = list(indices)

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

class DatasetWeightedSampler(object):
    def __init__(self, weights, sample_num):
        self.weights = weights
        self.sample_num = sample_num

    def sample(self):
        '''
        return:
            indices: data index list in dataset
        '''
        heap = []
        for idx, item in enumerate(self.weights):
            r = random.random()
            weight = r**(1/item)
            if idx < self.sample_num:
                heapq.heappush(heap, (weight, idx))
            else:
                heap_samll_weight, _ = heap[0]
                if weight > heap_samll_weight:
                    heapq.heappushpop(heap, (weight, idx))

            if idx % 10000 == 0:
                print('idx-{} completed...'.format(idx))

        indices = []
        for item in heap:
            indices.append(item[1])

        return indices
