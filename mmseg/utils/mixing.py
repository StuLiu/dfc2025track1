"""
@Project :
@File    : cutmix.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/12/25 15:07
@e-mail  : liuwa@hnu.edu.cn
"""
import numpy as np
import random
import cv2
import torch

import torch.nn.functional as tnf


def cutmix(inputs, targets, alpha=1.0):
    data_s, targets_s, data_t, targets_t = inputs.clone(), targets.clone(), inputs.clone(), targets.clone()
    shuffle_indices = torch.randperm(data_s.shape[0])
    data_t = data_t[shuffle_indices]
    targets_t = targets_t[shuffle_indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data_s.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data_s[:, :, y0:y1, x0:x1] = data_t[:, :, y0:y1, x0:x1]
    targets_s[:, y0:y1, x0:x1] = targets_t[:, y0:y1, x0:x1]
    return data_s, targets_s


def classmix(data_s, targets_s, ratio=0.5, class_num=9, ignore_label=255):
    """
    class mixing
    Args:
        data_s: [b, 3, h, w]
        targets_s: [b, 1, h, w]
        data_t: [b, 3, h, w]
        targets_t: [b, 1, h, w]
        ratio: 0~1
        class_num: number of classes

    Returns:
        mixed images and labels for both source and target domains
        data_s: [b, 3, h, w]
        targets_s: [b, h, w]
        data_t: [b, 3, h, w]
        targets_t: [b, h, w]
    """

    def index2onehot(labels, class_num=9, ignore_label=255):
        labels = labels.clone()
        if labels.dim() == 4:
            labels = labels.squeeze(dim=1)
        labels[labels == ignore_label] = class_num
        labels_onehot = tnf.one_hot(labels, num_classes=class_num + 1)[:, :, :, :-1]  # (b*h*w, c)
        labels_onehot = labels_onehot.permute(0, 3, 1, 2)
        return labels_onehot  # (b, c, h, w)

    data_s, targets_s, data_t, targets_t = (data_s.clone(), targets_s.clone().long(),
                                            data_s.clone(), targets_s.clone().long())
    b = data_t.shape[0]
    shuffle_indices = torch.randperm(b)
    data_s = data_s[shuffle_indices]
    targets_s = targets_s[shuffle_indices]

    if targets_s.dim() == 3:
        targets_s = targets_s.unsqueeze(dim=1)
    if targets_t.dim() == 3:
        targets_t = targets_t.unsqueeze(dim=1)

    class_ids = torch.randperm(class_num)[: int(class_num * ratio)]  # rand batch-wise
    # print(class_ids)
    class_mix = torch.zeros((b, class_num, 1, 1), dtype=torch.int).cuda()   # (1, c, 1, 1)
    for c_id in class_ids:
        class_mix[b // 2:, c_id, :, :] = 1

    targets_s_onehot = index2onehot(targets_s, class_num=class_num, ignore_label=ignore_label)
    cond_mix = torch.sum(targets_s_onehot * class_mix, dim=1, keepdim=True).bool()
    targets_t[cond_mix] = targets_s[cond_mix]
    cond_mix = torch.broadcast_to(cond_mix, data_t.shape)
    data_t[cond_mix] = data_s[cond_mix]

    return data_t, targets_t.squeeze(dim=1)


if __name__ == '__main__':
    image1 = cv2.imread('1.png')
    image1 = torch.Tensor(image1).cuda().int().permute(2, 0, 1).unsqueeze(dim=0)

    image2 = cv2.imread('2.png')
    image2 = torch.Tensor(image2).cuda().int().permute(2, 0, 1).unsqueeze(dim=0)

    image = torch.cat([image1, image2], dim=0)

    label1 = torch.zeros_like(image1[:, 0, :, :]).cuda()
    label1[:,:,:100] = 1
    label2 = torch.ones_like(image2[:, 0, :, :]).cuda()
    label2[:,:,:100] = 0
    label = torch.cat([label1, label2], dim=0)

    k = cv2.waitKey(1)

    while k != ord('q'):
        # img1, lbl1 = cutmix(image, label, alpha=1)
        img1, lbl1 = classmix(image, label, class_num=2, ignore_label=255)
        img1 = img1.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        lbl1 = lbl1.cpu().numpy().astype(np.uint8)
        cv2.imshow('i1', img1[0, :, :, :])
        cv2.imshow('i2', img1[1, :, :, :])
        cv2.imshow('l1', lbl1[0, :, :] * 255)
        cv2.imshow('l2', lbl1[1, :, :] * 255)

        k = cv2.waitKey(0)
