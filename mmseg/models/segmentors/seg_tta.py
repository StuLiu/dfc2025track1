# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Union

import os
# import ipdb
import numpy as np
from PIL import Image
import skimage.io as sio
import torch
import torch.nn as nn
from mmengine.model import BaseTTAModel
from mmengine.structures import PixelData

from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import SampleList
import warnings
import time


warnings.filterwarnings('ignore')


@MODELS.register_module()
class SegTTAModelV2(BaseTTAModel):
    def __init__(
        self,
        module: Union[dict, nn.Module],
        data_preprocessor: Union[dict, nn.Module, None] = None,
        save_mid_dir: str = None,
        save_mid_format: str = 'numpy',
    ):
        super().__init__(module, data_preprocessor)
        self.save_mid_dir = save_mid_dir
        self.save_mid_format = save_mid_format
        if self.save_mid_dir is not None and self.save_mid_dir != '':
            os.makedirs(save_mid_dir, exist_ok=True)

    def merge_preds(self, data_samples_list: List[SampleList]) -> SampleList:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[SampleList]): List of predictions
                of all enhanced data.

        Returns:
            SampleList: Merged prediction.
        """
        predictions = []
        for data_samples in data_samples_list:
            seg_logits = data_samples[0].seg_logits.data
            logits = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in data_samples:
                seg_logit = data_sample.seg_logits.data
                if self.module.out_channels > 1:
                    logits += seg_logit.softmax(dim=0)
                else:
                    logits += seg_logit.sigmoid()
            logits /= len(data_samples)
            if self.module.out_channels == 1:
                seg_pred = (logits > self.module.decode_head.threshold
                            ).to(logits).squeeze(1)
            else:
                seg_pred = logits.argmax(dim=0)

            # saving tta middle result
            if self.save_mid_dir is not None:
                img_name = os.path.basename(data_samples[0].img_path)[:-4]
                np.save(f'{self.save_mid_dir}/{img_name}.npy',
                        (logits.cpu().numpy() * 255).astype(np.uint8))
                # logits = (logits.cpu().numpy() * 255).astype(np.uint8)
                # for i in range(9):
                #     sio.imsave(f'{save_dir}/{img_name}_class_{i}.png', logits[i])

            # saved tta
            data_sample = SegDataSample(
                **{
                    'pred_sem_seg': PixelData(data=seg_pred),
                    'gt_sem_seg': data_samples[0].gt_sem_seg
                })
            data_sample.set_data({'pred_sem_seg': PixelData(data=seg_pred)})
            if hasattr(data_samples[0], 'gt_sem_seg'):
                data_sample.set_data(
                    {'gt_sem_seg': data_samples[0].gt_sem_seg})
            data_sample.set_metainfo({'img_path': data_samples[0].img_path})
            predictions.append(data_sample)
        return predictions


@MODELS.register_module()
class SegTTAModel(BaseTTAModel):

    def merge_preds(self, data_samples_list: List[SampleList]) -> SampleList:
        """Merge predictions of enhanced data to one prediction.

        Args:
            data_samples_list (List[SampleList]): List of predictions
                of all enhanced data.

        Returns:
            SampleList: Merged prediction.
        """
        predictions = []
        for data_samples in data_samples_list:
            seg_logits = data_samples[0].seg_logits.data
            logits = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in data_samples:
                seg_logit = data_sample.seg_logits.data
                if self.module.out_channels > 1:
                    logits += seg_logit.softmax(dim=0)
                else:
                    logits += seg_logit.sigmoid()
            logits /= len(data_samples)
            if self.module.out_channels == 1:
                seg_pred = (logits > self.module.decode_head.threshold
                            ).to(logits).squeeze(1)
            else:
                seg_pred = logits.argmax(dim=0)
            data_sample = SegDataSample(
                **{
                    'pred_sem_seg': PixelData(data=seg_pred),
                    'gt_sem_seg': data_samples[0].gt_sem_seg
                })
            data_sample.set_data({'pred_sem_seg': PixelData(data=seg_pred)})
            if hasattr(data_samples[0], 'gt_sem_seg'):
                data_sample.set_data(
                    {'gt_sem_seg': data_samples[0].gt_sem_seg})
            data_sample.set_metainfo({'img_path': data_samples[0].img_path})
            predictions.append(data_sample)
        return predictions

