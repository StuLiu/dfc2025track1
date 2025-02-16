'''
@Project : mmseg-agri 
@File    : colormaps.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2025/2/16 下午11:15
@e-mail  : 1183862787@qq.com
'''
import numpy as np
from mmseg.datasets import OpenEarthMapDataset

def get_palettes(dataset_name='oem'):
    """rgb format"""
    palette = None
    if dataset_name == 'oem':
        rgb_list = OpenEarthMapDataset.METAINFO['palette']
        palette = np.array(rgb_list).astype(np.uint8)
    return palette
