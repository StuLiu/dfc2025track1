# Obtained from: https://github.com/lhoyer/DAFormer
# Modifications:
# - Add img_interval
# - Add upscale_pred flag
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from copy import deepcopy
from collections import OrderedDict
from typing import List, Union, Dict
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from mmengine.optim import OptimWrapper

from torch import Tensor
from mmengine.model.wrappers import MMDistributedDataParallel
from mmengine.logging import print_log

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models import build_segmentor
from mmseg.registry import MODELS
from mmseg.utils import (ConfigType, OptConfigType, OptMultiConfig,
                         OptSampleList, SampleList, add_prefix)


def get_module(module):
    """Get `nn.ModuleDict` to fit the `MMDistributedDataParallel` interface.

    Args:
        module (MMDistributedDataParallel | nn.ModuleDict): The input
            module that needs processing.

    Returns:
        nn.ModuleDict: The ModuleDict of multiple networks.
    """
    if isinstance(module, DistributedDataParallel) or isinstance(module, MMDistributedDataParallel):
        return module.module

    return module


def ema(history, curr, alpha=0.999):
    return history * alpha + curr * (1 - alpha)


@MODELS.register_module()
class UDADecorator(BaseSegmentor):

    def __init__(self, segmentor_cfg, data_preprocessor):
        super(BaseSegmentor, self).__init__(data_preprocessor=data_preprocessor)
        segmentor_cfg['data_preprocessor'] = data_preprocessor
        self.model_stu = build_segmentor(deepcopy(segmentor_cfg))

        self.model_tea = deepcopy(self.model_stu)
        for param in self.model_tea.parameters():
            param.detach_()
        self.model_tea = self.model_tea.eval()

        self.train_cfg = segmentor_cfg['train_cfg']
        self.test_cfg = segmentor_cfg['test_cfg']

        self.iter = 0

    def get_student(self):
        return get_module(self.model_stu)

    def get_teacher(self):
        return get_module(self.model_tea)

    def extract_feat(self, inputs) -> List[Tensor]:
        """Extract features from images."""
        return self.get_student().extract_feat(inputs)

    @torch.no_grad()
    def extract_feat_tea(self, inputs) -> List[Tensor]:
        """Extract features from images."""
        return self.get_teacher().extract_feat(inputs)

    def encode_decode(self, inputs, batch_img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_student().encode_decode(inputs, batch_img_metas)

    @torch.no_grad()
    def encode_decode_tea(self, inputs, batch_img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        return self.get_teacher().encode_decode(inputs, batch_img_metas)

    def sync_ema_weights(self):
        ema_model = self.get_teacher()
        for param in ema_model.parameters():
            dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
            param.data /= dist.get_world_size()

    def update_ema_weights(self, curr_iter):
        alpha = min(1 - 1 / (curr_iter + 1), self.alpha)
        for ema_param, param in zip(self.get_teacher().parameters(),
                                    self.get_student().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = ema(ema_param.data, param.data, alpha)
            else:
                ema_param.data[:] = ema(ema_param[:].data[:],
                                        param[:].data[:], alpha)
        if dist.is_available() and dist.get_world_size() > 1 and curr_iter % 10 == 0:
            self.sync_ema_weights()

    def print_teacher_params(self):
        for name, param in self.get_teacher().named_parameters():
            if 'conv' in name:
                print_log(f'{name}: {param[0,0,:]}', None)
                break

    def _decode_head_forward_train(self, inputs: List[Tensor],
                                   data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        model_stu = self.get_student()
        loss_decode = model_stu.decode_head.loss(inputs, data_samples,
                                                 model_stu.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _auxiliary_head_forward_train(self, inputs: List[Tensor],
                                      data_samples: SampleList) -> dict:
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        model_stu = self.get_student()
        if isinstance(model_stu.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(model_stu.auxiliary_head):
                loss_aux = aux_head.loss(inputs, data_samples,
                                         model_stu.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = model_stu.auxiliary_head.loss(inputs, data_samples,
                                                     model_stu.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

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
        x = self.extract_feat(inputs)

        losses = dict()

        loss_decode = self._decode_head_forward_train(x, data_samples)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(x, data_samples)
            losses.update(loss_aux)
        # print('UDA_Decorator.loss(self, inputs: Tensor, data_samples: SampleList) -> dict ================')
        return losses

    def _forward(self,
                 inputs: Tensor,
                 data_samples: OptSampleList = None) -> Tensor:
        """Network forward process.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            Tensor: Forward output of model without any post-processes.
        """
        model_stu = self.get_student()
        x = model_stu.extract_feat(inputs)
        return model_stu.decode_head.forward(x)

    def predict(self,
                inputs: Tensor,
                data_samples: OptSampleList = None) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            inputs (Tensor): Inputs with shape (N, C, H, W).
            data_samples (List[:obj:`SegDataSample`], optional): The seg data
                samples. It usually includes information such as `metainfo`
                and `gt_sem_seg`.

        Returns:
            list[:obj:`SegDataSample`]: Segmentation results of the
            input images. Each SegDataSample usually contain:

            - ``pred_sem_seg``(PixelData): Prediction of semantic segmentation.
            - ``seg_logits``(PixelData): Predicted logits of semantic
                segmentation before normalization.
        """
        return self.get_student().predict(inputs, data_samples)

    def slide_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode with padding.
        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        return self.get_student().slide_inference(inputs, batch_img_metas)

    def whole_inference(self, inputs: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Inference with full image.

        Args:
            inputs (Tensor): The tensor should have a shape NxCxHxW, which
                contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        return self.get_student().whole_inference(inputs, batch_img_metas)

    def inference_tea(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        return self.get_teacher().inference(inputs, batch_img_metas)

    def inference(self, inputs: Tensor, batch_img_metas: List[dict]) -> Tensor:
        """Inference with slide/whole style.

        Args:
            inputs (Tensor): The input image of shape (N, 3, H, W).
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', 'pad_shape', and 'padding_size'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """
        return self.get_student().inference(inputs, batch_img_metas)

    def aug_test(self, inputs, batch_img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        return self.get_student().aug_test(inputs, batch_img_metas, rescale)

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
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

        print('UDA_Decorator.train_step(self, data...) -> Dict ================================')
        # # >>>>>>>>>>>>> official pipline begin: >>>>>>>>>>>>>>
        # # Enable automatic mixed precision training context.
        # with optim_wrapper.optim_context(self):
        #     data = self.data_preprocessor(data, True)
        #     losses = self._run_forward(data, mode='loss')  # type: ignore
        # parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        # optim_wrapper.update_params(parsed_losses)
        # return log_vars
        # # <<<<<<<<<<<<< official pipline end. <<<<<<<<<<<<<<<<
        raise NotImplementedError
