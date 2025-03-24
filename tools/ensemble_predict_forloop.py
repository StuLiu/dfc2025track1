"""
@Project : mmseg-agri
@File    : ensemble_weights.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/3/2 下午9:51
@e-mail  : 1183862787@qq.com
"""
# This script is utilized to prediction ensemble (weighted mean).
import os

import cv2
import numpy as np
import skimage.io as sio
import warnings
from tqdm import tqdm
from mmseg.utils.tools import render_segmentation_cv2
from mmseg.utils.palette import get_palettes

from mmengine.utils import track_parallel_progress, track_progress
from tqdm import tqdm


warnings.filterwarnings('ignore')


edge_mask_dir = './data/DFC2025Track1/test/edge_mask'
jc_name = 'test_ensemble'
size = (1024, 1024)
root_dir = './submits_mid/dfc_stage2'
# 定义ckpt文件列表和对应的权重
mid_dirs = [
    # # chusai
    # f'{root_dir}/02_09_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_sce-dice',           # 35.10
    # f'{root_dir}/02_14_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_gce-lovasz',         # 35.24
    # f'{root_dir}/02_17_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_ce-lovasz',          # 35.13
    # f'{root_dir}/02_17_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_sce-lovasz',         # 35.31
    f'{root_dir}/02_28_segformer_mit-b5_4xb2-160k_sarseg-896x896_stage1-28_ignore0_sce-lovasz',         # 35.70
    f'{root_dir}/02_29_segformer_mit-b5_4xb2-160k_sarseg-896x896_interv3_ignore0_sce-lovasz',           # 35.54
    f'{root_dir}/03_01_upernet_swinv2-base-w24_4xb2-160k_sarseg-768x768_interv3_ignore0_sce-lovasz',    # 35.57
    f'{root_dir}/03_01_upernet_convnextv2-base_4xb2-160k_sarseg-1024x1024_interv3_ignore0_sce-lovasz',  # 36.03
    # # fu sai
    # # the checkpoints are deleted along with the dockers in www.autodl.com
    # f'{root_dir}/03_07_segformer_mit-b5_4xb2-160k_sarseg-960x960_bestv2_ignore0_sce-lovasz',            # -
    # f'{root_dir}/03_07_upernet_convnextv2-base_4xb2-120k_sarseg-1024x1024_bestv2_ignore0_sce-lovasz',   # -
    # f'{root_dir}/03_07_upernet_convnextv2-large_4xb2-80k_sarseg-960x960_bestv2_ignore0_sce-lovasz',     # -
    # f'{root_dir}/03_07_upernet_convnextv2-large_4xb2-80k_sarseg-960x960_stage1-28_ignore0_sce-lovasz',  # 39.93
    # f'{root_dir}/03_07_upernet_swinv2-base-w24_4xb2-160k_sarseg-768x768_bestv2_ignore0_sce-lovasz',     # -
    # f'{root_dir}/03_07_upernet_swinv2-large-w24_4xb1-160k_sarseg-768x768_bestv2_ignore0_sce-lovasz',    # -
    # f'{root_dir}/03_07_upernet_swinv2-large-w24_4xb1-160k_sarseg-768x768_stage1-28_ignore0_sce-lovasz'  # -
]
weights = [1] * len(mid_dirs) # 每个模型的权重
weights_class = [
    0,      # Background
    1,      # Bareland
    1,      # Rangeland
    1,      # Developed space
    1,      # Road
    1,      # Tree
    1,      # Water
    1,      # Agriculture land
    1,      # Building
]
weights_class = np.array(weights_class)[:, np.newaxis, np.newaxis]
weights_class = np.broadcast_to(weights_class, (len(weights_class), *size))
assert len(mid_dirs) == len(weights) and len(mid_dirs) > 0
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
    logits = logits * weights_class
    out_ids = np.argmax(logits, axis=0).astype(np.uint8)

    # valid_mask = sio.imread(f'{edge_mask_dir}/{os.path.splitext(name)[0]}.png')
    valid_mask = cv2.imread(f'{edge_mask_dir}/{os.path.splitext(name)[0]}.png', flags=cv2.IMREAD_UNCHANGED)
    if np.sum(valid_mask) / valid_mask.shape[0] / valid_mask.shape[1] < 0.9985:
        out_ids = (out_ids * valid_mask).astype(np.uint8)
        print(f'{name}, {np.sum(valid_mask) / valid_mask.shape[0] / valid_mask.shape[1]: .4f}')

    # sio.imsave(f'{output_dir}/{os.path.splitext(name)[0]}.png', out_ids)
    cv2.imwrite(f'{output_dir}/{os.path.splitext(name)[0]}.png', out_ids)
    out_color = render_segmentation_cv2(out_ids, palette).astype(np.uint8)
    # sio.imsave(f'{output_dir_vis}/{os.path.splitext(name)[0]}.png', out_color)
    cv2.imwrite(f'{output_dir_vis}/{os.path.splitext(name)[0]}.png', out_color[:, :, ::-1])


# track_parallel_progress(process, file_names, 8)

# # If above progress do not work successfully, please try the single-thread progress below:
for file_name in tqdm(file_names):
    process(file_name)

print(f'final ensemble predictions is save at {output_dir}')
