"""
@Project : DFC2025track1
@File    : generate_zero_labels.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/1/27 下午9:11
@e-mail  : 1183862787@qq.com
"""


import os
import numpy as np
from skimage import io

def generate_zero_label(input_dir, output_dir):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif') or filename.endswith('.tif'):
            input_filepath = os.path.join(input_dir, filename)
            output_filepath = os.path.join(output_dir, filename)

            # 读取 TIFF 图像
            img = io.imread(input_filepath)

            # 创建一个全为 0 的数组，形状与图像的 shape 相同
            zero_label = np.ones_like(img, dtype=np.uint8)

            # 保存全为 0 的标签图像
            io.imsave(output_filepath, zero_label)

            print(f'Generated zero label for {filename} and saved to {output_filepath}')

# 示例用法
input_directory = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/test/sar_images'
output_directory = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/test/labels'

generate_zero_label(input_directory, output_directory)
