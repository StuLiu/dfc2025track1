"""
@Project : DFC2025track1
@File    : intersection_pesudo_labels.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/2/5 下午4:10
@e-mail  : 1183862787@qq.com
"""
import os
from os.path import join

import cv2
import numpy as np
import skimage.io as sio
from mmseg.utils.tools import render_segmentation_cv2
from mmseg.datasets import *
import warnings

warnings.filterwarnings('ignore')


def compute_intersection(label1, label2, ignore_label=0):
    """计算两个标签的交集"""
    mask = label1 != label2
    label_out = label2.copy()
    label_out[mask] = ignore_label
    return label_out


def process_labels(dir1, dir2, output_dir, dataset_name='DFC2025SarSegDataset', ignore_label=255):
    """处理两个目录下的标签文件"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir + '_color', exist_ok=True)

    palette = np.array(eval(dataset_name).METAINFO['palette']).astype(np.uint8)
    appends = (256 - len(palette)) * [[0, 0, 0]]
    palette = np.concatenate([palette, appends], axis=0)

    # 获取两个目录下的文件名列表
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    common_files = files1.intersection(files2)

    print(f"找到 {len(common_files)} 个相同文件名的标签文件。")
    idx = 0
    for file_name in common_files:
        # 读取两个标签文件
        label1_path = join(dir1, file_name)
        label2_path = join(dir2, file_name)
        label1 = sio.imread(label1_path)
        label2 = sio.imread(label2_path)

        # 确保两个标签的形状一致
        if label1.shape != label2.shape:
            print(f"文件 {file_name} 的标签形状不一致: {label1.shape} vs {label2.shape}")
            continue

        # 计算交集
        intersection = compute_intersection(label1, label2, ignore_label=ignore_label)
        labels_color = render_segmentation_cv2(intersection, palette)

        # 保存结果
        output_path = join(output_dir, file_name)
        sio.imsave(output_path, intersection)

        cv2.imwrite(output_dir + '_color/' + file_name[:-4] + '.png', labels_color[:, :, ::-1])
        idx += 1
        print(f"{idx}/{len(common_files)}, 已保存交集标签: {output_path}")


if __name__ == "__main__":
    # 定义两个输入目录和输出目录
    dir_official = ("/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/"
                    "02_26_uda_segformer_mit-b3_2xb4-80k_oem-768x768-alld_ignore255_dacsv2_ce_th0.968_downup_tta")  # 第二个标签目录
    dir_generated = ("/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/"
                     "02_28_uda_segformer_mit-b5_4xb2-80k_oem-896x896-alld_ignore255_dacsv2_ce_th0.968_downup_tta")  # 第一个标签目录
    dir_save = ("/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/"
                "stage1-28-29")  # 输出目录
    ignore_index = 0
    # 处理标签
    process_labels(dir_generated, dir_official, dir_save, ignore_label=ignore_index)
