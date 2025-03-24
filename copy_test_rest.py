'''
@Project : mmseg-agri 
@File    : copy_test_rest.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2025/3/23 上午2:29
@e-mail  : 1183862787@qq.com
'''
import os
import shutil


dir_npy = 'submits_mid/dfc_stage2/03_01_upernet_convnextv2-base_4xb2-160k_sarseg-1024x1024_interv3_ignore0_sce-lovasz'
dir_test = 'data/DFC2025Track1/test'
dir_test_rest = 'data/DFC2025Track1/test_rest'
os.makedirs(f'{dir_test_rest}/sar_images', exist_ok=True)
os.makedirs(f'{dir_test_rest}/labels', exist_ok=True)

names_test = os.listdir(dir_test + '/sar_images')
names_exiting = os.listdir(dir_npy)

for name_test in names_test:
    if f'{name_test[:-4]}.npy' not in names_exiting:
        shutil.copy(f'{dir_test}/sar_images/{name_test}', f'{dir_test_rest}/sar_images/{name_test}')
        shutil.copy(f'{dir_test}/labels/{name_test}', f'{dir_test_rest}/labels/{name_test}')
