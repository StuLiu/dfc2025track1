"""
@Project : mmseg-agri
@File    : dacs.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/2/15 下午1:30
@e-mail  : 1183862787@qq.com
"""
import math
import os
import cv2
import random
import numpy as np
import kornia
import torch
import torch.nn as nn
import torch.distributed as dist

from mmseg.utils.tools import img_tensor2cv2, render_segmentation_cv2
from mmseg.utils.palette import get_palettes

from torch import Tensor
from matplotlib import pyplot as plt
from copy import deepcopy
from typing import Union, Dict
from collections import OrderedDict

from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.logging import print_log
from mmengine.structures.pixel_data import PixelData
from mmengine.optim import OptimWrapper

from mmseg.models import build_segmentor
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from mmseg.models.segmentors.uda_decorator import UDADecorator, get_module


# from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
#                                                 get_mean_std, strong_transform)
# from mmseg.models.utils.visualization import subplotimg
# from mmseg.utils.utils import downscale_label_rati


def denorm_(img, mean, std):
    img.mul_(std).add_(mean).div_(255.0)


def renorm_(img, mean, std):
    img.mul_(255.0).sub_(mean).div_(std)


def color_jitter(color_jitter, mean, std, data=None, target=None, s=.25, p=.2):
    # s is the strength of colorjitter
    if not (data is None):
        if data.shape[1] == 3:
            if color_jitter < p:
                if isinstance(s, dict):
                    seq = nn.Sequential(kornia.augmentation.ColorJitter(**s))
                else:
                    seq = nn.Sequential(
                        kornia.augmentation.ColorJitter(
                            brightness=s, contrast=s, saturation=s, hue=s))
                denorm_(data, mean, std)
                data = seq(data)
                renorm_(data, mean, std)
    return data, target


def gaussian_blur(blur, data=None, target=None):
    if not (data is None):
        if data.shape[1] == 3:
            if blur > 0.5:
                sigma = np.random.uniform(0.15, 1.15)
                kernel_size_y = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[2]) - 0.5 +
                        np.ceil(0.1 * data.shape[2]) % 2))
                kernel_size_x = int(
                    np.floor(
                        np.ceil(0.1 * data.shape[3]) - 0.5 +
                        np.ceil(0.1 * data.shape[3]) % 2))
                kernel_size = (kernel_size_y, kernel_size_x)
                seq = nn.Sequential(
                    kornia.filters.GaussianBlur2d(
                        kernel_size=kernel_size, sigma=(sigma, sigma)))
                data = seq(data)
    return data, target


def cutmix(data_s, targets_s, data_t, targets_t, alpha=1.0):
    data_s, targets_s, data_t, targets_t = (data_s.clone(), targets_s.clone().squeeze(dim=1),
                                            data_t.clone(), targets_t.clone().squeeze(dim=1))
    # shuffle_indices = torch.randperm(data_s.shape[0])
    # data_t = data_t[shuffle_indices]
    # targets_t = targets_t[shuffle_indices]

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

    mask = torch.zeros_like(targets_s)
    if np.random.rand() < 0.5:
        data_t[:, :, y0:y1, x0:x1] = data_s[:, :, y0:y1, x0:x1]
        targets_t[:, y0:y1, x0:x1] = targets_s[:, y0:y1, x0:x1]
        mask[:, y0:y1, x0:x1] = 1
        return data_t, targets_t, mask
    else:
        data_s[:, :, y0:y1, x0:x1] = data_t[:, :, y0:y1, x0:x1]
        targets_s[:, y0:y1, x0:x1] = targets_t[:, y0:y1, x0:x1]
        mask[:, y0:y1, x0:x1] = 1
        mask = 1 - mask
        return data_s, targets_s, mask


@MODELS.register_module()
class CutMix(UDADecorator):

    def __init__(
            self,
            segmentor,
            data_preprocessor,
            alpha=0.999,
            pseudo_threshold=0.968,
            ignore_index=255,
            pseudo_weight_ignore_margin=(0, 0, 0, 0),  # top, bottom, left, right
            imnet_model_cfg=None,
            imnet_feature_dist_lambda=0,
            imnet_feature_dist_classes=None,
            imnet_feature_dist_scale_min_ratio=None,
            mix='class',
            blur=True,
            color_jitter_strength=0.2,
            color_jitter_probability=0.2,
            debug_img_interval=1000,
            palette='oem',  # default is open earth map
            print_grad_magnitude=False,
            debug=False,
    ):
        super(CutMix, self).__init__(segmentor, data_preprocessor)
        self.debug = debug
        self.local_iter = 0
        self.alpha = alpha

        self.pseudo_threshold = pseudo_threshold
        self.ignore_index = ignore_index
        self.pseudo_weight_ignore_margin = pseudo_weight_ignore_margin

        self.fdist_lambda = imnet_feature_dist_lambda
        self.fdist_classes = imnet_feature_dist_classes
        self.fdist_scale_min_ratio = imnet_feature_dist_scale_min_ratio
        self.enable_fdist = self.fdist_lambda > 0

        self.mix = mix
        self.blur = blur
        self.color_jitter_s = color_jitter_strength
        self.color_jitter_p = color_jitter_probability

        self.debug_img_interval = debug_img_interval
        self.palette = palette
        self.print_grad_magnitude = print_grad_magnitude
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}

        if self.enable_fdist:
            if imnet_model_cfg is None:
                self.imnet_model = build_segmentor(deepcopy(segmentor_cfg))
            else:
                self.imnet_model = build_segmentor(deepcopy(imnet_model_cfg))
        else:
            self.imnet_model = None

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper):
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # sync ema teacher
        if (self.iter % 50 == 0):
            if dist.is_initialized():
                dist.barrier()
            self.sync_ema_weights()

        self.iter += 1
        # update teacher params
        self.update_ema_weights(self.iter)
        self.get_teacher().eval()

        log_vars = {}

        # forward -> loss.backword -> optimizer.step -> return loss logs.
        with optim_wrapper.optim_context(self):
            # preprocess batch data
            data_copy = deepcopy(data)
            data_src = {
                'inputs': data_copy['inputs'],
                'data_samples': data_copy['data_samples']
            }
            data_tgt = {
                'inputs': data_copy['inputs_tgt'],
                'data_samples': data_copy['data_samples_tgt']
            }
            data_src = self.data_preprocessor(data_src, training=True)
            data_tgt = self.data_preprocessor(data_tgt, training=True)

            # ###################### #
            # train on source domain #
            # ###################### #
            losses_src = self._run_forward(data_src, mode='loss')
            losses_parsed_src, log_vars_src = self.parse_losses(losses_src)
            for loss_name, loss_val in log_vars_src.items():
                log_vars[f'{loss_name}_src'] = loss_val
            optim_wrapper.update_params(losses_parsed_src)
            log_vars.update(log_vars_src)

            # ###################### #
            # train on target domain #
            # ###################### #
            # get mean std and strong-aug params
            batch_size = data_src['inputs'].size(0)
            metainfo_0 = data_src['data_samples'][0].metainfo
            means, stds = self.get_mean_std(metainfo_0, batch_size, data_src['inputs'].device)
            strong_parameters = {
                'mix': None,
                'color_jitter': random.uniform(0, 1),
                'color_jitter_s': self.color_jitter_s,
                'color_jitter_p': self.color_jitter_p,
                'blur': random.uniform(0, 1) if self.blur else 0,
                'mean': means[0].unsqueeze(dim=0),  # assume same normalization
                'std': stds[0].unsqueeze(dim=0)
            }

            # get pseudo labels
            imgs_tgt_origin = data_tgt['inputs'].clone()
            with torch.no_grad():
                seg_logits = self.get_teacher().predict_logits(data_tgt['inputs'], data_tgt['data_samples']).detach()
            seg_map_teacher = seg_logits.argmax(dim=1)
            pseudo_label, pseudo_weight = self.get_pseudo_label_and_weight(seg_logits)
            # pseudo_weight = self.filter_valid_pseudo_region(pseudo_weight)

            imgs_src = data_src['inputs']
            lbls_src = self._stack_batch_gt(data_src['data_samples'])
            imgs_tgt, pseudo_label, mix_masks = cutmix(imgs_src, lbls_src, imgs_tgt_origin, pseudo_label)
            pseudo_weight[mix_masks == 1] = 1

            # apply strong augs
            imgs_tgt, pseudo_label = self.strong_transform(
                strong_parameters,
                data=imgs_tgt,
                target=pseudo_label
            )
            data_tgt['inputs'] = imgs_tgt
            data_tgt['data_samples'] = self._revert_batch_gt(data_tgt['data_samples'], pseudo_label)

            # forward and backward
            losses_tgt = self._run_forward(data_tgt, mode='loss')
            for loss_name in losses_tgt.keys():
                if 'loss_ce' in loss_name:
                    losses_tgt[loss_name] = losses_tgt[loss_name] * pseudo_weight
                if 'loss_lovasz' in loss_name or 'loss_dice' in loss_name:
                    losses_tgt[loss_name] = losses_tgt[loss_name] * pseudo_weight.min(dim=1)[0].min(dim=1)[0]
            losses_parsed_tgt, log_vars_tgt = self.parse_losses(losses_tgt)
            for loss_name, loss_val in log_vars_tgt.items():
                log_vars[f'{loss_name}_tgt'] = loss_val
            optim_wrapper.update_params(losses_parsed_tgt)
            log_vars.update(log_vars_tgt)

        if self.debug and (self.iter == 1 or self.iter % self.debug_img_interval == 0):
            self.print_teacher_params()
            print_log(f'dist.is_initialized()={dist.is_initialized()}', 'current')
            print_log(f'pseudo_weight[0].unique()={torch.unique(pseudo_weight[0])}', 'current')
            print_log(f'{(seg_logits.softmax(dim=1)[0] >= self.pseudo_threshold).sum() / pseudo_weight[0].numel()}',
                      'current')
            if not dist.is_initialized() or dist.get_rank() == 0:
                work_dir = os.path.join(self.train_cfg.get('work_dir', 'debugs'), 'debug')
                os.makedirs(work_dir, exist_ok=True)

                img_src_0 = img_tensor2cv2(data_src['inputs'][0, :3, :, :].cpu())
                lbl_src_0_vis = lbls_src[0].squeeze().cpu().numpy().astype(np.uint8)
                lbl_src_0_vis = render_segmentation_cv2(lbl_src_0_vis, get_palettes(self.palette))[:, :, ::-1]

                img_tgt_0_origin = img_tensor2cv2(imgs_tgt_origin[0, :3, :, :].cpu())
                pseudo_label_0_vis = seg_map_teacher[0].squeeze().cpu().numpy().astype(np.uint8)
                pseudo_label_0_vis = render_segmentation_cv2(pseudo_label_0_vis, get_palettes(self.palette))[:, :, ::-1]

                img_tgt_0 = img_tensor2cv2(data_tgt['inputs'][0, :3, :, :].cpu())
                lbl_tgt_0_vis = pseudo_label[0].squeeze().cpu().numpy().astype(np.uint8)
                lbl_tgt_0_vis = render_segmentation_cv2(lbl_tgt_0_vis, get_palettes(self.palette))[:, :, ::-1]

                mix_masks_0 = (mix_masks[0].squeeze().cpu().numpy() * 255).astype(np.uint8)
                mix_masks_0 = np.stack([mix_masks_0] * 3, axis=-1)
                mixed_seg_weight_0 = (pseudo_weight[0].squeeze().cpu().numpy() * 255).astype(np.uint8)
                mixed_seg_weight_0 = np.stack([mixed_seg_weight_0] * 3, axis=-1)

                img_src_0 = cv2.resize(img_src_0, dsize=(448, 448), interpolation=cv2.INTER_AREA)
                img_tgt_0_origin = cv2.resize(img_tgt_0_origin, dsize=(448, 448), interpolation=cv2.INTER_AREA)
                img_tgt_0 = cv2.resize(img_tgt_0, dsize=(448, 448), interpolation=cv2.INTER_AREA)
                lbl_src_0_vis = cv2.resize(lbl_src_0_vis, dsize=(448, 448), interpolation=cv2.INTER_NEAREST)
                lbl_tgt_0_vis = cv2.resize(lbl_tgt_0_vis, dsize=(448, 448), interpolation=cv2.INTER_NEAREST)
                pseudo_label_0_vis = cv2.resize(pseudo_label_0_vis, dsize=(448, 448), interpolation=cv2.INTER_NEAREST)
                mix_masks_0 = cv2.resize(mix_masks_0, dsize=(448, 448), interpolation=cv2.INTER_NEAREST)
                mixed_seg_weight_0 = cv2.resize(mixed_seg_weight_0, dsize=(448, 448), interpolation=cv2.INTER_NEAREST)

                imgs = np.concatenate([img_src_0, img_tgt_0_origin, img_tgt_0, mix_masks_0], axis=1)
                lbls = np.concatenate([lbl_src_0_vis, pseudo_label_0_vis, lbl_tgt_0_vis, mixed_seg_weight_0],
                                      axis=1)
                vis = np.concatenate([imgs, lbls], axis=0)
                cv2.imwrite(os.path.join(work_dir, f'vis_{self.iter:06d}.png'), vis)
                # cv2.imshow("vis", vis)
                # if cv2.waitKey(0) == ord('q'):
                #     exit(0)

        return log_vars
        # <<<<<<<<<<<<< dacs training pipline end. <<<<<<<<<<<<<<<<

    def get_pseudo_label_and_weight(self, logits):
        ema_softmax = torch.softmax(logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        valid_num = torch.sum(pseudo_prob >= self.pseudo_threshold, dim=[1, 2], keepdim=True)   # (n, 1, 1)
        pseudo_weight = valid_num / pseudo_label[0].numel()             # (n, 1, 1)
        pseudo_weight = pseudo_weight * torch.ones_like(pseudo_prob)    # (n, h, w)
        return pseudo_label, pseudo_weight

    def filter_valid_pseudo_region(self, pseudo_weight, valid_pseudo_mask=None):
        # Don't trust pseudo-labels in regions with potential
        # rectification artifacts. This can lead to a pseudo-label
        # drift from sky towards building or traffic light.
        t, b, l, r = self.pseudo_weight_ignore_margin
        pseudo_weight[:, :t, :] = 0
        pseudo_weight[:, -b:, :] = 0
        pseudo_weight[:, :, :l] = 0
        pseudo_weight[:, :, -r:] = 0

        if valid_pseudo_mask is not None:
            pseudo_weight *= valid_pseudo_mask.squeeze(1)
        return pseudo_weight

    @staticmethod
    def get_class_masks(labels):
        def generate_class_mask(label_, classes_):
            label_, classes_ = torch.broadcast_tensors(label_,
                                                       classes_.unsqueeze(1).unsqueeze(2))
            class_mask_ = label_.eq(classes_).sum(0, keepdims=True)
            return class_mask_

        class_masks = []
        for label in labels:
            classes = torch.unique(labels)
            nclasses = classes.shape[0]
            class_choice = np.random.choice(
                nclasses, int((nclasses + nclasses % 2) / 2), replace=False)
            classes = classes[torch.Tensor(class_choice).long()]
            class_masks.append(generate_class_mask(label, classes).unsqueeze(0))
        return class_masks

    @staticmethod
    def get_mean_std(metainfo, batch_size, dev):
        mean = [
            torch.as_tensor(metainfo['mean'], device=dev)
            for _ in range(batch_size)
        ]
        mean = torch.stack(mean).view(-1, 3, 1, 1)
        std = [
            torch.as_tensor(metainfo['std'], device=dev)
            for _ in range(batch_size)
        ]
        std = torch.stack(std).view(-1, 3, 1, 1)
        return mean, std

    @staticmethod
    def strong_transform(param, data=None, target=None):
        assert ((data is not None) or (target is not None))
        data, target = color_jitter(
            color_jitter=param['color_jitter'],
            s=param['color_jitter_s'],
            p=param['color_jitter_p'],
            mean=param['mean'],
            std=param['std'],
            data=data,
            target=target)
        data, target = gaussian_blur(blur=param['blur'], data=data, target=target)
        return data, target

    @staticmethod
    def _stack_batch_gt(data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    @staticmethod
    def _revert_batch_gt(data_samples: SampleList, gts: Tensor) -> SampleList:
        data_samples_copy = deepcopy(data_samples)
        for data_sample, gt in zip(data_samples_copy, gts):
            gt_sem_seg = PixelData(data=gt.squeeze(dim=0))
            data_sample.gt_sem_seg = gt_sem_seg
        return data_samples_copy
