'''
@Project : mmseg-agri 
@File    : generate_results_from_mid.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2024/5/24 上午10:05
@e-mail  : 1183862787@qq.com
'''
import os
import numpy as np
import skimage.io as sio
import warnings
from tqdm import tqdm


warnings.filterwarnings('ignore')

results_dir = '/mnt/home/liuwang_data/results_mid/deeplabv3plus_swin-large_rc_mosaic4x_tta_51.33'
test_img_dir = '/mnt/home/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/img_dir/test'
test_imgs = os.listdir(test_img_dir)
save_dir ="/mnt/home/liuwang_data/results_enc_dec/deeplabv3plus_swin-large_rc_mosaic4x_tta_51.33"
os.makedirs(save_dir,exist_ok=True)

for img_name in tqdm(test_imgs):

    results_list = []
    for i in range(9):
        img_path = os.path.join(results_dir, f'{img_name[:-4]}_class_{i}.png')
        img = sio.imread(img_path)
        results_list.append(img)
    pred = np.stack(results_list, axis=0)
    pred = np.argmax(pred, axis=0)
    sio.imsave(os.path.join(save_dir, f'{img_name[:-4]}.png'), pred.astype(np.uint8))
