'''
@Project : mmseg-agri 
@File    : cpuntest.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2025/2/21 下午2:32
@e-mail  : 1183862787@qq.com
'''
import os
import shutil


dir_src = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/rgb_images'
dir_tgt = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/rgb_images_temp'
os.makedirs(dir_tgt, exist_ok=False)
dir_src_lbl = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels'
dir_tgt_lbl = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_temp'
os.makedirs(dir_tgt_lbl, exist_ok=False)

dir_pl = (f'/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/train/labels_pl/'
          f'02_26_uda_segformer_mit-b3_2xb4-80k_oem-768x768-alld_ignore255_dacsv2_ce_th0.968_downup_tta')

names = os.listdir(dir_src)
names_pl = os.listdir((dir_pl))

for name in names:
    if name.replace('.tif', '.png') not in names_pl:
        shutil.copy(f'{dir_src}/{name}', f'{dir_tgt}/{name}')
        shutil.copy(f'{dir_src_lbl}/{name}', f'{dir_tgt_lbl}/{name}')
        # break