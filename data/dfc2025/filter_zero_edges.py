import cv2
import numpy as np
import skimage.io as sio
import os


def detect_invalid_regions(image_path, output_mask_path):
    # 加载图像（以灰度模式加载）
    img = sio.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"图像文件 {image_path} 不存在或无法加载.")

    # 创建初始掩码图像（大小与输入图像相同，初始值为0）
    mask = np.zeros_like(img, dtype=np.uint8)

    # 找出图像中全黑的区域，这些区域可能是无效区域
    # 将全黑像素定义为阈值0以下的区域
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓以获取边界
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有的轮廓
    for contour in contours:
        # 拟合直线（直线上可能是无效区域与有效区域的分界线）
        # 如果点数小于2，无法拟合直线，直接跳过
        if contour.shape[0] < 2:
            continue

        # 使用最小二乘法拟合轮廓（直线）
        rows, cols = img.shape
        [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        # vx 和 vy 是直线方向向量，x0 和 y0 是直线上的点

        # 根据直线方向填充掩码图像
        slope = vy / vx if vx != 0 else float('inf')

        # 确定无效区域：填充轮廓内部为1
        cv2.drawContours(mask, [contour], -1, color=255, thickness=cv2.FILLED)

    # 将掩码图像中无效区域设置为1，其他区域保持为0
    mask = (mask > 0).astype(np.uint8)

    # 保存掩码图像
    cv2.imwrite(output_mask_path, mask * 255)
    print(f"保存掩码图像到 {output_mask_path}.")


# 使用示例
if __name__ == "__main__":
    root_dir = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/DFC2025Track1/test/sar_images'

    out_dir = f'{os.path.dirname(root_dir)}/edge_mask'
    os.makedirs(out_dir, exist_ok=True)

    # img_name = "TestArea_087.tif"  # 输入图像路径
    names = os.listdir(root_dir)
    for img_name in names:
        detect_invalid_regions(f'{root_dir}/{img_name}', f'{out_dir}/{img_name}'[:-4]+'.png')
