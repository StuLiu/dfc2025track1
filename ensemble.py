import os
import numpy as np
import skimage.io as sio
from PIL import Image

results_dir = ["./05_15_SegFormer-B3-RGBN-ACWLoss-Mosaicx4-TTA",
               "./05_17_UperNet-ConvNextV1-Large-RGBN-ACWLoss-Mosaicx4-TTA", 
               "./05_22_SegFormer-B3-RGBN-ACWLoss-Mosaicx4-Mosaicx9-TTA"]
test_img_dir = "/mnt/home/wangzhiyu_data/Data/Challenge/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/img_dir/test/"
test_imgs = os.listdir(test_img_dir)

save_dir = "./ensemble_results"
os.makedirs(save_dir, exist_ok=True)
for test_img in test_imgs:
    results_list = []
    for i in range(9):
        result = np.zeros((1, 512, 512))
        for dir in results_dir:
            result_file = f"{dir}/{test_img[:-4]}_class_{i}.png"
            result += np.array(Image.open(result_file))[None, ...]
        results_list.append(result)

    results = np.concatenate(results_list, 0)
    results = np.argmax(results, 0)
    sio.imsave(f"{save_dir}/{test_img[:-4]}.png", results.astype(np.uint8))
