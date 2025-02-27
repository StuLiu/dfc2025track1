'''
@Project : mmseg-agri 
@File    : convert_png2tif.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2025/2/22 上午12:28
@e-mail  : 1183862787@qq.com
'''
import skimage.io as sio
import os
import warnings
from tqdm import tqdm
import argparse

warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser(description='Train a segmentor')
parser.add_argument('--dir-name',
                    default='02_20_uda_segformer_mit-b3_4xb2-80k_oem-768x768-alld_ignore255_dacs_ce_th0.968_tta',
                    help='train config file path')
parser.add_argument('--work-dir', help='the dir to save logs and models')
args = parser.parse_args()

dir_path = (f'/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/'
            f'{args.dir_name}')
names = os.listdir(dir_path)
for name in tqdm(names):
    img = sio.imread(f'{dir_path}/{name}')
    sio.imsave(f'{dir_path}/{name.split(".png")[0]}.tif', img)
