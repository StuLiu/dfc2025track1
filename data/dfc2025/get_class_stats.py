# Obtained from: https://github.com/lhoyer/DAFormer
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import argparse
import json
import os.path as osp

import mmcv
import numpy as np
import skimage.io as sio
import mmengine


def get_sample_status(file):
    # re-assign labels to match the format of Cityscapes
    label = sio.imread(file)
    sample_class_stats = {}
    for cid in np.unique(label):
        k_mask = label == cid
        n = int(np.sum(k_mask))
        sample_class_stats[int(cid)] = n
    sample_class_stats['file'] = file
    return sample_class_stats


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert GTA annotations to TrainIds')
    parser.add_argument('--data-path', help='data path',
                        default='/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train')
    parser.add_argument('--gt-dir', type=str, default='labels')
    parser.add_argument('--out-dir', help='output path')
    parser.add_argument('--postfix', help='.png, .tif, or others', default='.tif')
    parser.add_argument(
        '--nproc', default=8, type=int, help='number of process')
    args = parser.parse_args()
    return args


def save_class_stats(out_dir, sample_class_stats):
    with open(osp.join(out_dir, 'sample_class_stats.json'), 'w') as of:
        json.dump(sample_class_stats, of, indent=2)

    sample_class_stats_dict = {}
    for stats in sample_class_stats:
        f = stats.pop('file')
        sample_class_stats_dict[f] = stats
    with open(osp.join(out_dir, 'sample_class_stats_dict.json'), 'w') as of:
        json.dump(sample_class_stats_dict, of, indent=2)

    samples_with_class = {}
    for file, stats in sample_class_stats_dict.items():
        for c, n in stats.items():
            if c not in samples_with_class:
                samples_with_class[c] = [(file, n)]
            else:
                samples_with_class[c].append((file, n))
    with open(osp.join(out_dir, 'samples_with_class.json'), 'w') as of:
        json.dump(samples_with_class, of, indent=2)


def main():
    args = parse_args()
    data_path = args.data_path
    out_dir = args.out_dir if args.out_dir else data_path
    mmengine.mkdir_or_exist(out_dir)

    gt_dir = osp.join(data_path, args.gt_dir)

    poly_files = []
    for poly in mmengine.utils.scandir(
            gt_dir, suffix=args.postfix, recursive=True):
        poly_file = osp.join(str(gt_dir), str(poly))
        poly_files.append(poly_file)
    poly_files = sorted(poly_files)

    only_postprocessing = False
    if not only_postprocessing:
        if args.nproc > 1:
            sample_class_stats = mmengine.utils.track_parallel_progress(
                get_sample_status, poly_files, args.nproc)
        else:
            sample_class_stats = mmengine.utils.track_progress(
                get_sample_status, poly_files)
    else:
        with open(osp.join(out_dir, 'sample_class_stats.json'), 'r') as of:
            sample_class_stats = json.load(of)

    save_class_stats(out_dir, sample_class_stats)


if __name__ == '__main__':
    main()
