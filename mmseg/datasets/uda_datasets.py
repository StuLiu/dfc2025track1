"""
@Project : mmseg-agri
@File    : uda_datasets.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/2/15 下午3:45
@e-mail  : 1183862787@qq.com
"""
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import json
import os.path as osp

import mmcv
import numpy as np
import torch

from . import CityscapesDataset
from mmengine.registry.build_functions import build_from_cfg
from mmengine.registry import DATASETS
from mmengine.logging import print_log


def get_rcs_class_probs(data_root, temperature, ignore_indexes=()):
    with open(osp.join(data_root, 'sample_class_stats.json'), 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)
    for idx in ignore_indexes:
        freq[idx] = 0
    freq = freq / (freq.sum() + 1e-7)
    return list(overall_class_stats.keys()), freq.numpy()


@DATASETS.register_module()
class UDADataset(object):

    def __init__(self,
                 source,
                 target,
                 rare_class_sampling=None):
        self.source = DATASETS.build(source)
        self.target = DATASETS.build(target)
        self.METAINFO = self.target.METAINFO
        assert self.target.METAINFO['classes'] == self.source.METAINFO['classes']
        assert self.target.METAINFO['palette'] == self.source.METAINFO['palette']

        self.rcs_enabled = rare_class_sampling is not None
        if self.rcs_enabled:
            assert source['serialize_data'] == False
            self.rcs_class_temp = rare_class_sampling['class_temp']
            self.rcs_min_crop_ratio = rare_class_sampling['min_crop_ratio']
            self.rcs_min_pixels = rare_class_sampling['min_pixels']
            self.ignore_indexes = rare_class_sampling['ignore_indexes']

            self.rcs_classes, self.rcs_classprob = get_rcs_class_probs(
                data_root=source['data_root'],
                temperature=self.rcs_class_temp,
                ignore_indexes=self.ignore_indexes
            )
            print_log(f'RCS Classes: {self.rcs_classes}', 'current')
            print_log(f'RCS ClassProb: {self.rcs_classprob}', 'current')

            with open(osp.join(source['data_root'], 'samples_with_class.json'),
                      'r') as of:
                samples_with_class_and_n = json.load(of)
            samples_with_class_and_n = {
                int(k): v
                for k, v in samples_with_class_and_n.items()
                if int(k) in self.rcs_classes
            }
            self.samples_with_class = {}
            for c in self.rcs_classes:
                self.samples_with_class[c] = []
                for file, pixels in samples_with_class_and_n[c]:
                    if pixels > self.rcs_min_pixels:
                        self.samples_with_class[c].append(file.split('/')[-1])
                assert len(self.samples_with_class[c]) > 0
            self.file_to_idx = {}
            for i, dic in enumerate(self.source.data_list):
                file = dic['seg_map_path']
                file = file.split('/')[-1]
                self.file_to_idx[file] = i
            pass

    def get_rare_class_sample(self):
        c = np.random.choice(self.rcs_classes, p=self.rcs_classprob)
        f1 = np.random.choice(self.samples_with_class[c])
        i1 = self.file_to_idx[f1]
        s1 = self.source[i1]
        if self.rcs_min_crop_ratio > 0:
            for j in range(10):
                n_class = torch.sum(s1['data_samples'].gt_sem_seg.data == c)
                # mmcv.print_log(f'{j}: {n_class}', 'mmseg')
                if n_class > self.rcs_min_pixels * self.rcs_min_crop_ratio:
                    break
                # Sample a new random crop from source image i1.
                # Please note, that self.source.__getitem__(idx) applies the
                # preprocessing pipeline to the loaded image, which includes
                # RandomCrop, and results in a new crop of the image.
                s1 = self.source[i1]
        i2 = np.random.choice(range(len(self.target)))
        s2 = self.target[i2]

        return s1, s2

    def __getitem__(self, idx):
        if self.rcs_enabled:
            s1, s2 = self.get_rare_class_sample()
        else:
            s1 = self.source[idx // len(self.target)]
            s2 = self.target[idx % len(self.target)]
        out_dict = {
            'inputs': s1['inputs'],
            'data_samples': s1['data_samples'],
            'inputs_tgt': s2['inputs'],
            'data_samples_tgt': s2['data_samples'],
        }
        return out_dict

    def __len__(self):
        return len(self.source) * len(self.target)
