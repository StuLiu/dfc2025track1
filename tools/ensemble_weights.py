"""
@Project : mmseg-agri
@File    : ensemble_weights.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/3/2 下午9:51
@e-mail  : 1183862787@qq.com
"""
import os
import torch
from collections import OrderedDict


# 定义ckpt文件列表和对应的权重
ckpt_files = [
    'work_dirs/02_09_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_sce-dice/iter_160000.pth',     #35.1
    'work_dirs/02_14_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_gce-lovasz/iter_160000.pth',   #35.24
    'work_dirs/02_17_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_ce-lovasz/iter_160000.pth',    #35.13
    'work_dirs/02_17_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_sce-lovasz/iter_160000.pth',   #35.31
    # 'work_dirs/02_28_segformer_mit-b5_4xb2-160k_sarseg-896x896_stage1-28_ignore0_sce-lovasz/iter_160000.pth',   #35.70
    # 'work_dirs/02_29_segformer_mit-b5_4xb2-160k_sarseg-896x896_interv3_ignore0_sce-lovasz/iter_160000.pth'      #35.57
]
weights = [0.25, 0.25, 0.25, 0.25]  # 每个模型的权重，确保权重总和为1
output_file = 'ckpts_mid/segformer_mit-b5_56.pth'  # 替换为实际的输出文件路径

# 检查权重总和是否为1
assert sum(weights) == 1.0, "权重总和必须为1"

os.makedirs(os.path.dirname(output_file), exist_ok=True)
ckpt_out = torch.load(ckpt_files[0])

# 初始化集成后的权重字典
integrated_state_dict = None

# 遍历每个ckpt文件，进行加权集成
for idx, (weight, ckpt_file) in enumerate(zip(weights, ckpt_files)):
    # 加载模型权重
    state_dict = torch.load(ckpt_file)['state_dict']

    # 如果是第一个文件，直接初始化集成后的权重
    if integrated_state_dict is None:
        integrated_state_dict = OrderedDict()
        for key, value in state_dict.items():
            integrated_state_dict[key] = weight * value
    else:
        # 对于后续文件，将权重按比例累加
        for key, value in state_dict.items():
            integrated_state_dict[key] += weight * value
    print(f"parseing {idx}/{len(ckpt_files)}...")

# 保存集成后的权重到指定文件
ckpt_out['state_dict'] = integrated_state_dict
torch.save(ckpt_out, output_file)
print(f"集成后的权重已保存到 {output_file}")
