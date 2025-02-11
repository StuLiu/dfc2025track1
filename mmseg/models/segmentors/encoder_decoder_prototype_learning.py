# Copyright (c) OpenMMLab. All rights reserved.
import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Optional, Dict

from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)
from .base import BaseSegmentor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.models.utils.prototype_dist_estimator import Prototypes
from mmengine.structures.pixel_data import PixelData
from mmengine.logging import print_log


@MODELS.register_module()
class EncoderDecoderPrototypeLearning(EncoderDecoder):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.

    1. The ``loss`` method is used to calculate the loss of model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2) Call the decode head loss function to forward decode head model and
    calculate losses.

    .. code:: text

     loss(): extract_feat() -> _decode_head_forward_train() -> _auxiliary_head_forward_train (optional)
     _decode_head_forward_train(): decode_head.loss()
     _auxiliary_head_forward_train(): auxiliary_head.loss (optional)

    2. The ``predict`` method is used to predict segmentation results,
    which includes two steps: (1) Run inference function to obtain the list of
    seg_logits (2) Call post-processing function to obtain list of
    ``SegDataSample`` including ``pred_sem_seg`` and ``seg_logits``.

    .. code:: text

     predict(): inference() -> postprocess_result()
     infercen(): whole_inference()/slide_inference()
     whole_inference()/slide_inference(): encoder_decoder()
     encoder_decoder(): extract_feat() -> decode_head.predict()

    3. The ``_forward`` method is used to output the tensor by running the model,
    which includes two steps: (1) Extracts features to obtain the feature maps
    (2)Call the decode head forward function to forward decode head model.

    .. code:: text

     _forward(): extract_feat() -> _decode_head.forward()

    Args:

        backbone (ConfigType): The config for the backnone of segmentor.
        decode_head (ConfigType): The config for the decode head of segmentor.
        neck (OptConfigType): The config for the neck of segmentor.
            Defaults to None.
        auxiliary_head (OptConfigType): The config for the auxiliary head of
            segmentor. Defaults to None.
        train_cfg (OptConfigType): The config for training. Defaults to None.
        test_cfg (OptConfigType): The config for testing. Defaults to None.
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        pretrained (str, optional): The path for pretrained model.
            Defaults to None.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.
    """  # noqa: E501

    def __init__(self,
                 backbone: ConfigType,
                 decode_head: ConfigType,
                 neck: OptConfigType = None,
                 auxiliary_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 pretrained: Optional[str] = None,
                 init_cfg: OptMultiConfig = None,
                 prototype_cfg: Dict = None,
                 warmup_iters: int = 0,
                 loss_weight_proto: float = 1,
                 threshold: float = 0.968
                 ):
        super().__init__(backbone, decode_head, neck, auxiliary_head, train_cfg, test_cfg,
                         data_preprocessor, pretrained, init_cfg)
        if prototype_cfg is not None:
            self.prototype = Prototypes(**prototype_cfg)
        else:
            self.prototype = None

        self.warmup_iters = warmup_iters if isinstance(warmup_iters, int) else 0
        self.iter = 0
        self.loss_weight_proto = loss_weight_proto
        self.threshold = threshold
        self.debug = prototype_cfg.get('debug', False)

    @staticmethod
    def _stack_batch_gt(data_samples: SampleList) -> Tensor:
        gt_semantic_segs = [
            data_sample.gt_sem_seg.data for data_sample in data_samples
        ]
        return torch.stack(gt_semantic_segs, dim=0)

    @staticmethod
    def _revert_batch_gt(data_samples: SampleList, gts: Tensor) -> SampleList:
        data_samples_copy = copy.deepcopy(data_samples)
        for data_sample, gt in zip(data_samples_copy, gts):
            gt_sem_seg = PixelData(data=gt)
            data_sample.gt_sem_seg = gt_sem_seg
        return data_samples_copy

    def _prototype_training(self, data_samples, annotations, seg_logits: List[Tensor], features: Tensor) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()

        # generate pseudo_labels
        # pseudo_labels = seg_logits.argmax(dim=1)
        cosine_sim = self.prototype.similarity_cosine(features)
        cosine_sim = F.interpolate(cosine_sim, size=annotations.shape[-2:], mode='bilinear', align_corners=False)
        pseudo_labels = cosine_sim.argmax(dim=1).unsqueeze(dim=1)
        pseudo_labels[pseudo_labels != annotations] = self.prototype.ignore_index

        if self.debug and self.iter % 500 == 0:
            percent_valide = (pseudo_labels != self.prototype.ignore_index).sum() / pseudo_labels.numel()
            print(f'valid percentage {(pseudo_labels != self.prototype.ignore_index).sum()} '
                  f'/ {pseudo_labels.numel()} = {percent_valide * 100:.2f}%')

        data_samples = self._revert_batch_gt(data_samples, pseudo_labels)
        loss_decode_proto = self.decode_head.loss_by_feat(seg_logits, data_samples)

        # re-weight the prototype loss
        for k in loss_decode_proto.keys():
            if 'loss' in k:
                loss_decode_proto[k] = loss_decode_proto[k] * self.loss_weight_proto

        losses.update(loss_decode_proto)
        return losses

    def loss(self, inputs: Tensor, data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            inputs (Tensor): Input images.
            data_samples (list[:obj:`SegDataSample`]): The seg data samples.
                It usually includes information such as `metainfo` and
                `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        annotations = self._stack_batch_gt(data_samples)

        # forward backbone
        feats = self.extract_feat(inputs)

        # forward seg head
        seg_logits = self.decode_head.forward(feats)

        # update prototypes
        self.prototype.update(feats[-1], annotations, seg_logits)

        # decode main loss for segmentation
        loss_decode_main = self.decode_head.loss_by_feat(seg_logits, data_samples)
        losses.update(add_prefix(loss_decode_main, 'decode'))

        if self.iter >= self.warmup_iters:
            # decode prototypical learning loss for segmentation
            loss_decode_proto = self._prototype_training(data_samples, annotations, seg_logits, feats[-1])
            losses.update(add_prefix(loss_decode_proto, 'decode.proto'))

        # the same as the origin auxiliary decode head
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(feats, data_samples)
            losses.update(loss_aux)

        self.iter += 1
        return losses

    # def _prototype_training(self, data_samples, seg_logits: List[Tensor], features: Tensor) -> dict:
    #     """Run forward function and calculate loss for decode head in
    #     training."""
    #     losses = dict()
    #
    #     # generate pseudo_labels
    #     # pseudo_labels = seg_logits.argmax(dim=1)
    #     cosine_sim = self.prototype.similarity_cosine(features)
    #     cosine_sim = F.interpolate(cosine_sim, size=seg_logits.shape[-2:], mode='bilinear', align_corners=False)
    #     cosine_sim_softmax = torch.softmax(cosine_sim, dim=1)
    #     max_probs, pseudo_labels = cosine_sim_softmax.max(dim=1)
    #     pseudo_labels[max_probs < self.threshold] = self.prototype.ignore_index
    #
    #     if self.debug and self.iter % 500 == 0:
    #         percent_valide = (max_probs >= self.threshold).sum() / pseudo_labels.numel()
    #         print(f'valid percentage {(max_probs >= self.threshold).sum()} '
    #               f'/ {pseudo_labels.numel()} = {percent_valide * 100:.2f}%')
    #         thres = [0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1]
    #         for th in thres:
    #             percent_valide = (max_probs >= th).sum() / pseudo_labels.numel()
    #             print(f'{th:1.3f}\t : valid percentage {(max_probs >= th).sum()} '
    #                   f'/ {pseudo_labels.numel()} = {percent_valide * 100:.2f}%')
    #
    #     data_samples = self._revert_batch_gt(data_samples, pseudo_labels)
    #     loss_decode_proto = self.decode_head.loss_by_feat(seg_logits, data_samples)
    #
    #     # re-weight the prototype loss
    #     for k in loss_decode_proto.keys():
    #         if 'loss' in k:
    #             loss_decode_proto[k] = loss_decode_proto[k] * self.loss_weight_proto
    #
    #     losses.update(loss_decode_proto)
    #     return losses
