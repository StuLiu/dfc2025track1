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

warnings.filterwarnings('ignore')

dir_name = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/' \
           '02_20_uda_segformer_mit-b3_4xb2-80k_oem-768x768-alld_ignore255_dacs_ce_th0.968_tta'
names = os.listdir(dir_name)
for name in tqdm(names):
    img = sio.imread(f'{dir_name}/{name}')
    sio.imsave(f'{dir_name}/{name.split(".png")[0]}.tif', img)
