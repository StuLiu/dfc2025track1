"""
@Project : mmseg-agri
@File    : ensemble_weights.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/3/2 下午9:51
@e-mail  : 1183862787@qq.com
"""
# This script is utilized to ensemble soft pseudo labels.
import os
import numpy as np
import skimage.io as sio
import warnings
from tqdm import tqdm
from mmseg.utils.tools import render_segmentation_cv2
from mmseg.utils.palette import get_palettes

from mmengine.utils import track_parallel_progress, track_progress


warnings.filterwarnings('ignore')


jc_name = 'stage1_28_30_31_32_ensemble'
size = (1024, 1024)
root_dir = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl_mid'
postfix = '.tif'
# 定义ckpt文件列表和对应的权重
mid_dirs = [
    f'{root_dir}/02_26_uda_segformer_mit-b3_2xb4-80k_oem-768x768-alld_ignore255_dacsv2_ce_th0.968_downup',
    f'{root_dir}/03_02_uda_upernet_swinv2-base_4xb1-80k_oem-1024x1024-alld_ignore255_dacsv2_ce_th0.968_downup',
    f'{root_dir}/03_02_uda_segformer_mit-b5_4xb1-80k_oem-1024x1024-alld_ignore255_dacsv2_ce_th0.968_downup',
    f'{root_dir}/03_02_uda_upernet_convnext-base_4xb1-80k_oem-1024x1024-alld_ignore255_dacsv2_ce_th0.968_downup',
]
weights = [1, 1, 1, 1]
output_dir = f'{root_dir}/{jc_name}'
output_dir_vis = f'{output_dir}_color'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir_vis, exist_ok=True)
palette = get_palettes('oem')

file_names = os.listdir(mid_dirs[0])


def process(name):
    logits = 0
    for idx, mid_dir in enumerate(mid_dirs):
        logits_tmp = np.load(f'{mid_dir}/{name}').astype(np.int32)
        logits = logits + logits_tmp * weights[idx]
    out_ids = np.argmax(logits, axis=0).astype(np.uint8)
    sio.imsave(f'{output_dir}/{os.path.splitext(name)[0]}{postfix}', out_ids)
    out_color = render_segmentation_cv2(out_ids, palette).astype(np.uint8)
    sio.imsave(f'{output_dir_vis}/{os.path.splitext(name)[0]}.png', out_color)

# track_progress(process, file_names)
track_parallel_progress(process, file_names, 16)
