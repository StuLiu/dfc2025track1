"""
@Project : mmseg-agri
@File    : multi_class_acw_loss.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2024/4/24 下午1:26
@e-mail  : 1183862787@qq.com
"""
# Copyright (c) OpenMMLab. All rights reserved.
import warnings
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

from mmseg.registry import MODELS


def soft_jaccard_score(output: torch.Tensor, target: torch.Tensor,
                       smooth: float = 0.0, eps: float = 1e-7, dims=None) -> torch.Tensor:
    assert output.size() == target.size()
    mask = (torch.sum(target, dim=1, keepdim=True) > 0).float()     # pixel mask
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(mask * (output + target), dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(mask * (output + target))

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)

    return jaccard_score


@MODELS.register_module()
class MultiLabelJaccardLoss(nn.Module):

    def __init__(
            self,
            classes: Optional[List[int]] = None,
            log_loss: bool = False,
            from_logits: bool = True,
            smooth: float = 0.,
            eps: float = 1e-7,
            mean=True,
            rare_class_sampling=False,
            ini_weight=0, ini_iteration=0,
            use_sigmoid=False, loss_weight=1.0
    ):
        """Implementation of Jaccard loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`, otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            ignore_index: Label that indicates ignored pixels (does not contribute to loss)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **y_pred** - torch.Tensor of shape (N, C, H, W)
             - **y_true** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        super(MultiLabelJaccardLoss, self).__init__()

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        self.mean = mean
        self.rcs = rare_class_sampling
        self.loss_weight = loss_weight
        self.loss_name = 'loss_jcd'

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, weight=None, ignore_index=255) -> torch.Tensor:

        assert y_true.size(0) == y_pred.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            y_pred = F.logsigmoid(y_pred).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        dims = (0, 2)

        y_true = y_true.view(bs, num_classes, -1)
        if self.rcs:
            # for new iou
            y_true1 = y_true * torch.sum(y_true, dim=1, keepdim=True)
            # for unbalance data
            y_true1[:, 1, :] += y_true[:, 1, :] * 1
            y_true1[:, 3, :] += y_true[:, 3, :] * 1
            y_true1[:, 5, :] += y_true[:, 5, :] * 1
            y_true1[:, 8, :] += y_true[:, 8, :] * 1

            y_true = y_true1.clone()

        y_pred = y_pred.view(bs, num_classes, -1)


        score = soft_jaccard_score(y_pred, y_true.type(y_pred.dtype), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(score.clamp_min(self.eps))
        else:
            loss = 1.0 - score

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        if self.classes is not None:
            loss = loss[self.classes]

        if self.mean:
            return loss.sum() / (mask.float().sum() + self.eps)

        return self.loss_weight * loss


@MODELS.register_module()
class MultiLabelACWLoss(nn.Module):
    def __init__(self, ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255):
        super(MultiLabelACWLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps
        self._loss_name = "loss_mlacw"
        self.loss_weight = loss_weight
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.jcd_criterion = MultiLabelJaccardLoss(log_loss=False, from_logits=False, mean=True)

    def forward(self, prediction, target, weight=None, ignore_index=-100):
        # return self.loss_weight * self.forward_sgcnet(prediction, target)
        return self.loss_weight * self.forward_sgcnet(prediction, target)

    def forward_new(self, prediction, target, weight=None, ignore_index=-100):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, C, H, W) ground truth
        return:  loss_acw
        """
        assert prediction.shape == target.shape and weight is None
        mask = (torch.sum(target, dim=1, keepdim=True) > 0).float()

        pred = F.logsigmoid(prediction).exp()       # sigmoid

        _, acw_pixel = self.adaptive_class_weight(pred, target)

        loss_bce = mask * acw_pixel * self.bce_criterion(prediction, target.float())

        loss_jcd = self.jcd_criterion(pred, target)

        return loss_bce.mean() + loss_jcd

    def forward_sgcnet(self, prediction, target, weight=None, ignore_index=-100):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, C, H, W) ground truth
        return:  loss_acw
        """
        assert prediction.shape == target.shape and weight is None

        mask = (torch.sum(target, dim=1, keepdim=True) > 0).float()

        pred = F.logsigmoid(prediction).exp()       # sigmoid

        _, acw_pixel = self.adaptive_class_weight(pred, target)

        err = torch.pow((target - pred), 2)
        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = mask * acw_pixel * pnc

        intersection = 2 * torch.sum(mask * pred * target, dim=(0, 2, 3)) + self.eps
        union = mask * (pred + target)
        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        # print(loss_pnc.mean(), - dice.mean().log())
        return torch.sum(loss_pnc, dim=1).mean() - dice.mean().log()

    def adaptive_class_weight(self, pred, target):
        self.itr += 1

        sum_class = torch.sum(target, dim=(0, 2, 3))
        weight_curr = sum_class / (sum_class.sum() + self.eps)

        self.weight = (self.weight * (self.itr - 1) + weight_curr) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / (mfb.sum() + self.eps)

        acw_class = mfb
        acw_pixel = (1. + pred.detach() + target) * mfb.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return acw_class, acw_pixel

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


@MODELS.register_module()
class MultiLabelBCELoss(nn.Module):
    def __init__(self, ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255):
        super(MultiLabelBCELoss, self).__init__()
        self.ignore_index = ignore_index
        self._loss_name = "loss_mlbce"
        self.loss_weight = loss_weight
        self.bce_criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, prediction, target, weight=None, ignore_index=-100):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, C, H, W) ground truth
        return:  loss_acw
        """
        assert prediction.shape == target.shape and weight is None

        mask = (torch.sum(target, dim=1, keepdim=True) > 0).long()

        # avoid loss is float
        if mask.max() < 1:
            mask = 1e-7

        loss_bce = mask * self.bce_criterion(prediction, target.float())

        return self.loss_weight * torch.sum(loss_bce, dim=1).mean()

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name



if __name__ == '__main__':
    loss_mlbce = MultiLabelBCELoss()
    loss_mljcd = MultiLabelJaccardLoss(mean=False)
    loss_mlacw = MultiLabelACWLoss()

    a = torch.randint(-10, 10, (2, 2, 512, 512)).float().cuda()
    a[:, 0, :, :] = -10000
    a[:, 1, :, :] = 10000
    b = torch.randint(0, 2, (2, 2, 512, 512)).long().cuda()
    b[:, 0, :, :] = 1
    b[:, 1, :, :] = 0

    print(loss_mlbce(a, b))
    print(loss_mljcd(a, b))
    print(loss_mlacw(a, b))