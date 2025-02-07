# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from mmseg.registry import MODELS
from .utils import get_class_weight, weight_reduce_loss
from .dice_loss import DiceLoss


def cross_entropy(pred,
                  label,
                  weight=None,
                  class_weight=None,
                  reduction='mean',
                  avg_factor=None,
                  ignore_index=-100,
                  avg_non_ignore=False):
    """cross_entropy. The wrapper function for :func:`F.cross_entropy`

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
            Default: None.
        class_weight (list[float], optional): The weight for each class.
            Default: None.
        reduction (str, optional): The method used to reduce the loss.
            Options are 'none', 'mean' and 'sum'. Default: 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Default: None.
        ignore_index (int): Specifies a target value that is ignored and
            does not contribute to the input gradients. When
            ``avg_non_ignore `` is ``True``, and the ``reduction`` is
            ``''mean''``, the loss is averaged over non-ignored targets.
            Defaults: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    # class_weight is a manual rescaling weight given to each class.
    # If given, has to be a Tensor of size C element-wise losses
    loss = F.cross_entropy(
        pred,
        label,
        weight=class_weight,
        reduction='none',
        ignore_index=ignore_index)

    # apply weights and do the reduction
    # average loss over non-ignored elements
    # pytorch's official cross_entropy average loss over non-ignored elements
    # refer to https://github.com/pytorch/pytorch/blob/56b43f4fec1f76953f15a627694d4bba34588969/torch/nn/functional.py#L2660  # noqa
    if (avg_factor is None) and reduction == 'mean':
        if class_weight is None:
            if avg_non_ignore:
                avg_factor = label.numel() - (label
                                              == ignore_index).sum().item()
            else:
                avg_factor = label.numel()

        else:
            # the average factor should take the class weights into account
            label_weights = torch.stack([class_weight[cls] for cls in label
                                         ]).to(device=class_weight.device)

            if avg_non_ignore:
                label_weights[label == ignore_index] = 0
            avg_factor = label_weights.sum()

    if weight is not None:
        weight = weight.float()
    loss = weight_reduce_loss(
        loss, weight=weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def _expand_onehot_labels(labels, label_weights, target_shape, ignore_index):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_zeros(target_shape)
    valid_mask = (labels >= 0) & (labels != ignore_index)
    inds = torch.nonzero(valid_mask, as_tuple=True)

    if inds[0].numel() > 0:
        if labels.dim() == 3:
            bin_labels[inds[0], labels[valid_mask], inds[1], inds[2]] = 1
        else:
            bin_labels[inds[0], labels[valid_mask]] = 1

    valid_mask = valid_mask.unsqueeze(1).expand(target_shape).float()

    if label_weights is None:
        bin_label_weights = valid_mask
    else:
        bin_label_weights = label_weights.unsqueeze(1).expand(target_shape)
        bin_label_weights = bin_label_weights * valid_mask

    return bin_labels, bin_label_weights, valid_mask


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None,
                         ignore_index=-100,
                         avg_non_ignore=False,
                         **kwargs):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
            Note: In bce loss, label < 0 is invalid.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (int): The label index to be ignored. Default: -100.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.size(1) == 1:
        # For binary class segmentation, the shape of pred is
        # [N, 1, H, W] and that of label is [N, H, W].
        # As the ignore_index often set as 255, so the
        # binary class label check should mask out
        # ignore_index
        assert label[label != ignore_index].max() <= 1, \
            'For pred with shape [N, 1, H, W], its label must have at ' \
            'most 2 classes'
        pred = pred.squeeze(1)
    if pred.dim() != label.dim():
        assert (pred.dim() == 2 and label.dim() == 1) or (
                pred.dim() == 4 and label.dim() == 3), \
            'Only pred shape [N, C], label shape [N] or pred shape [N, C, ' \
            'H, W], label shape [N, H, W] are supported'
        # `weight` returned from `_expand_onehot_labels`
        # has been treated for valid (non-ignore) pixels
        label, weight, valid_mask = _expand_onehot_labels(
            label, weight, pred.shape, ignore_index)
    else:
        # should mask out the ignored elements
        valid_mask = ((label >= 0) & (label != ignore_index)).float()
        if weight is not None:
            weight = weight * valid_mask
        else:
            weight = valid_mask
    # average loss over non-ignored and valid elements
    if reduction == 'mean' and avg_factor is None and avg_non_ignore:
        avg_factor = valid_mask.sum().item()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), pos_weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


def mask_cross_entropy(pred,
                       target,
                       label,
                       reduction='mean',
                       avg_factor=None,
                       class_weight=None,
                       ignore_index=None,
                       **kwargs):
    """Calculate the CrossEntropy loss for masks.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction.
        label (torch.Tensor): ``label`` indicates the class label of the mask'
            corresponding object. This will be used to select the mask in the
            of the class which the object belongs to when the mask prediction
            if not class-agnostic.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.
        ignore_index (None): Placeholder, to be consistent with other loss.
            Default: None.

    Returns:
        torch.Tensor: The calculated loss
    """
    assert ignore_index is None, 'BCE loss does not support ignore_index'
    # TODO: handle these two reserved arguments
    assert reduction == 'mean' and avg_factor is None
    num_rois = pred.size()[0]
    inds = torch.arange(0, num_rois, dtype=torch.long, device=pred.device)
    pred_slice = pred[inds, label].squeeze(1)
    return F.binary_cross_entropy_with_logits(
        pred_slice, target, weight=class_weight, reduction='mean')[None]


class SCELoss(torch.nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, labels_one_hot):
        """
        Args:
            pred: softmaxed logits
            labels_one_hot: one-hot label

        Returns:

        """
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        # CCE
        ce = -1 * labels_one_hot * torch.log(pred)

        # RCE
        labels_one_hot = torch.clamp(labels_one_hot, min=1e-4, max=1.0)
        rce = -1 * pred * torch.log(labels_one_hot)

        # Loss
        loss = self.alpha * ce + self.beta * rce
        return loss

@MODELS.register_module()
class ACWSCELoss(nn.Module):
    def __init__(self,  ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255):
        super(ACWSCELoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps
        self.sce = SCELoss(alpha=1.0, beta=1.0)
        self._loss_name = "loss_acw"

    def forward(self, prediction, target, weight=None, ignore_index=-100):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        assert weight is None
        pred = F.softmax(prediction, 1)
        # print(target.unique())
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        # acw-sce
        sce = self.sce(pred, one_hot_label)
        loss_sce = torch.sum(acw * sce, 1)

        # Dice
        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        return loss_sce.mean() - dice.mean().log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()

        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        if mask is not None:
            acw[mask] = 0

        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None

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
class ACWLoss(nn.Module):
    def __init__(self,  ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255):
        super(ACWLoss, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps
        self._loss_name = "loss_acw"

    def forward(self, prediction, target, weight=None, ignore_index=-100):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        assert weight is None
        pred = F.softmax(prediction, 1)
        # print(target.unique())
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
        # one = torch.ones_like(err)

        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)


        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        return loss_pnc.mean() - dice.mean().log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()

        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        if mask is not None:
            acw[mask] = 0

        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None

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
class ACWJaccardLoss(ACWLoss):

    def __init__(self, ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255, smooth=100):
        super().__init__(ini_weight, ini_iteration, eps, use_sigmoid, loss_weight, ignore_index)
        self.smooth = smooth
        self._loss_name = 'acwj_loss'

    def forward(self, prediction, target, weight=None, ignore_index=-100):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        assert weight is None
        pred = F.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        # pnc loss
        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)

        # jaccard
        intersection = torch.sum(pred * one_hot_label, dim=(0, 2, 3))

        cardinality = pred + one_hot_label
        if mask is not None:
            cardinality[mask] = 0
        cardinality = torch.sum(cardinality, dim=(0, 2, 3))

        union = cardinality - intersection
        jaccard = (intersection + self.eps) / (union + self.eps)

        return loss_pnc.mean() - jaccard.mean().log()


@MODELS.register_module()
class ACWDefocalLoss(ACWLoss):

    def __init__(self, ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255, smooth=100):
        super().__init__(ini_weight, ini_iteration, eps, use_sigmoid, loss_weight, ignore_index)
        self.smooth = smooth
        self._loss_name = 'acwdf_loss'

    def forward(self, prediction, target, weight=None, ignore_index=-100):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        assert weight is None
        pred = F.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        # defocal weight: loss larger, weight smaller
        df_weight = 1.0 / (torch.abs(one_hot_label - pred.detach()) + self.eps)   # (n, c, h, w)
        df_weight /= df_weight.mean()

        # pnc loss
        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(df_weight * acw * pnc, 1)

        # jaccard
        intersection = torch.sum(pred * one_hot_label, dim=(0, 2, 3))

        cardinality = pred + one_hot_label
        if mask is not None:
            cardinality[mask] = 0
        cardinality = torch.sum(cardinality, dim=(0, 2, 3))

        union = cardinality - intersection
        jaccard = (intersection + self.eps) / (union + self.eps)

        return loss_pnc.mean() - jaccard.mean().log()



@MODELS.register_module()
class ACWFocalLoss(ACWLoss):

    def __init__(self, ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255, smooth=100, gamma=2.0):
        super().__init__(ini_weight, ini_iteration, eps, use_sigmoid, loss_weight, ignore_index)
        self.smooth = smooth
        self.gamma = gamma
        self._loss_name = 'acwf_loss'

    def forward(self, prediction, target, weight=None, ignore_index=-100):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        assert weight is None
        pred = F.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        # defocal weight: loss larger, weight smaller
        f_weight = torch.pow(torch.abs(one_hot_label - pred.detach()), self.gamma)   # (n, c, h, w)
        f_weight /= f_weight.mean()

        # pnc loss
        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(f_weight * acw * pnc, 1)

        # jaccard
        intersection = torch.sum(pred * one_hot_label, dim=(0, 2, 3))

        cardinality = pred + one_hot_label
        if mask is not None:
            cardinality[mask] = 0
        cardinality = torch.sum(cardinality, dim=(0, 2, 3))

        union = cardinality - intersection
        jaccard = (intersection + self.eps) / (union + self.eps)

        return loss_pnc.mean() - jaccard.mean().log()


@MODELS.register_module()
class ACWLossV2(nn.Module):
    def __init__(self,  ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255):
        super(ACWLossV2, self).__init__()
        self.ignore_index = ignore_index
        self.weight = ini_weight
        self.weight_cls = ini_weight
        self.itr = ini_iteration
        self.eps = eps
        self._loss_name = "loss_acw2"

    def forward(self, prediction, target, weight=None, ignore_index=-100):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        assert weight is None
        pred = F.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        acw, acw_cls = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
        # one = torch.ones_like(err)

        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)


        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        return loss_pnc.mean() - (acw_cls * dice).mean().log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()

        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / (mfb.sum() + self.eps)
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        ones_cls = (sum_class > 0).long()
        self.weight_cls = (self.weight_cls * (self.itr - 1) + ones_cls) / self.itr
        acw_cls = self.weight_cls.mean() / (self.weight_cls + self.eps)

        if mask is not None:
            acw[mask] = 0

        # if self.itr % 50 == 1:
        #     print(acw_cls)

        return acw, acw_cls

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None

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



class Dice(nn.Module):
    def __init__(self, aux_weights: list = [1, 0.4, 0.4], ignore_label=255):
        """
        delta: Controls weight given to FP and FN. This equals to dice score when delta=0.5
        """
        super().__init__()
        self.aux_weights = aux_weights
        self.ignore_label = ignore_label
        self.eps = 1e-7

    def _forward(self, preds: Tensor, labels: Tensor) -> Tensor:
        # preds in shape [B, C, H, W] and labels in shape [B, H, W]
        labels = labels.clone()
        num_classes = preds.shape[1]
        labels[labels == self.ignore_label] = num_classes
        labels = F.one_hot(labels, num_classes + 1)[:, :, :, :-1]  # (b, h, w, c)

        pre = preds.softmax(dim=1)
        pre = pre.permute(1, 0, 2, 3).reshape(num_classes, -1)     # (c, -1)
        lbl = labels.permute(3, 0, 1, 2).reshape(num_classes, -1)            # (c, -1)

        tp = torch.sum(pre * lbl, dim=1)            # (c,)
        fn = torch.sum(lbl * (1 - pre), dim=1)      # (c,)
        fp = torch.sum((1 - lbl) * pre, dim=1)      # (c,)

        dice_score = (2 * tp + self.eps) / (2 * tp + fn + fp + self.eps)
        dice_loss = torch.mean(1 - dice_score)

        return dice_loss

    def forward(self, preds, targets: Tensor) -> Tensor:
        if isinstance(preds, tuple):
            return sum([w * self._forward(pred, targets) for (pred, w) in zip(preds, self.aux_weights)])
        return self._forward(preds, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean', ignore_label=255):
        """
        Focal Loss for semantic segmentation.

        Parameters:
        - alpha: Balancing factor for class imbalance.
        - gamma: Focusing parameter to reduce the effect of easy samples.
        - reduction: Specifies the reduction to apply to the output: 'none', 'mean' or 'sum'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_label = ignore_label

    def forward(self, inputs, targets):
        """
        Compute the focal loss.

        Parameters:
        - inputs: Predicted logits (before softmax), shape (N, C, H, W)
        - targets: Ground truth labels (C) as long tensor, shape (N, H, W)
        """
        # 计算 softmax 得到预测的概率分布
        bce_loss = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_label)(inputs, targets)
        p_t = torch.exp(-bce_loss)  # 预测为正确类的概率

        # 计算 focal loss
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # 'none'



@MODELS.register_module()
class HybridV1(nn.Module):
    # CE + Dice
    def __init__(self, ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255):
        super().__init__()
        self.ignore_label = ignore_index
        self.criterion_ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.criterion_dice = Dice(ignore_label=ignore_index)
        self._loss_name = 'loss_hybridv1'

    def forward(self, preds, labels, weight=None, ignore_index=255):
        loss_ce = self.criterion_ce(preds, labels)
        loss_dice = self.criterion_dice(preds, labels)
        return loss_ce + loss_dice

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
class HybridV2(nn.Module):
    # focal + Dice
    def __init__(self, ini_weight=0, ini_iteration=0, eps=1e-5,
                 use_sigmoid=False, loss_weight=1.0, ignore_index=255):
        super().__init__()
        self.ignore_label = ignore_index
        self.criterion_focal = FocalLoss(ignore_label=ignore_index)
        self.criterion_dice = Dice(ignore_label=ignore_index)
        self._loss_name = 'loss_hybridv2'

    def forward(self, preds, labels, weight=None, ignore_index=255):
        loss_focal = self.criterion_focal(preds, labels)
        loss_dice = self.criterion_dice(preds, labels)
        return loss_focal + loss_dice

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

# @MODELS.register_module()
# class HybridV1(nn.Module):
#     """
#     CE + dice
#     """
#     def __init__(self,  ini_weight=0, ini_iteration=0, eps=1e-5,
#                  use_sigmoid=False, loss_weight=1.0, ignore_index=255):
#         super(HybridV1, self).__init__()
#         self.ignore_index = ignore_index
#         self.weight = ini_weight
#         self.itr = ini_iteration
#         self.eps = eps
#         self._loss_name = "loss_cedice"
#         self.ce_critetion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
#
#     def forward(self, prediction, target, weight=None, ignore_index=-100):
#         """
#         pred :    shape (N, C, H, W)
#         target :  shape (N, H, W) ground truth
#         return:  loss_acw
#         """
#         assert weight is None
#
#         ce_loss = self.ce_critetion(prediction, target)
#
#         pred = F.softmax(prediction, 1)
#         # print(target.unique())
#         one_hot_label, mask = self.encode_one_hot_label(pred, target)
#
#         intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
#         union = pred + one_hot_label
#
#         if mask is not None:
#             union[mask] = 0
#
#         union = torch.sum(union, dim=(0, 2, 3)) + self.eps
#         dice = intersection / union
#
#         return ce_loss - dice.mean().log()
#
#     def encode_one_hot_label(self, pred, target):
#         one_hot_label = pred.detach() * 0
#         if self.ignore_index is not None:
#             mask = (target == self.ignore_index)
#             target = target.clone()
#             target[mask] = 0
#             one_hot_label.scatter_(1, target.unsqueeze(1), 1)
#             mask = mask.unsqueeze(1).expand_as(one_hot_label)
#             one_hot_label[mask] = 0
#             return one_hot_label, mask
#         else:
#             one_hot_label.scatter_(1, target.unsqueeze(1), 1)
#             return one_hot_label, None
#
#     @property
#     def loss_name(self):
#         """Loss Name.
#
#         This function must be implemented and will return the name of this
#         loss function. This name will be used to combine different loss items
#         by simple sum operation. In addition, if you want this loss item to be
#         included into the backward graph, `loss_` must be the prefix of the
#         name.
#
#         Returns:
#             str: The name of this loss item.
#         """
#         return self._loss_name
#


@MODELS.register_module()
class CrossEntropyLoss(nn.Module):
    """CrossEntropyLoss.

    Args:
        use_sigmoid (bool, optional): Whether the prediction uses sigmoid
            of softmax. Defaults to False.
        use_mask (bool, optional): Whether to use mask cross entropy loss.
            Defaults to False.
        reduction (str, optional): . Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        class_weight (list[float] | str, optional): Weight of each class. If in
            str format, read them from a file. Defaults to None.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
        loss_name (str, optional): Name of the loss item. If you want this loss
            item to be included into the backward graph, `loss_` must be the
            prefix of the name. Defaults to 'loss_ce'.
        avg_non_ignore (bool): The flag decides to whether the loss is
            only averaged over non-ignored targets. Default: False.
            `New in version 0.23.0.`
    """

    def __init__(self,
                 use_sigmoid=False,
                 use_mask=False,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_ce',
                 avg_non_ignore=False,
                 ignore_index=255):
        super().__init__()
        assert (use_sigmoid is False) or (use_mask is False)
        self.use_sigmoid = use_sigmoid
        self.use_mask = use_mask
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.class_weight = get_class_weight(class_weight)
        self.avg_non_ignore = avg_non_ignore
        self.ignore_index = ignore_index
        if not self.avg_non_ignore and self.reduction == 'mean':
            warnings.warn(
                'Default ``avg_non_ignore`` is False, if you would like to '
                'ignore the certain label and average loss over non-ignore '
                'labels, which is the same with PyTorch official '
                'cross_entropy, set ``avg_non_ignore=True``.')

        if self.use_sigmoid:
            self.cls_criterion = binary_cross_entropy
        elif self.use_mask:
            self.cls_criterion = mask_cross_entropy
        else:
            self.cls_criterion = cross_entropy
        self._loss_name = loss_name

    def extra_repr(self):
        """Extra repr."""
        s = f'avg_non_ignore={self.avg_non_ignore}'
        return s

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=-100,
                **kwargs):
        """Forward function."""
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.class_weight is not None:
            class_weight = cls_score.new_tensor(self.class_weight)
        else:
            class_weight = None
        # Note: for BCE loss, label < 0 is invalid.
        loss_cls = self.loss_weight * self.cls_criterion(
            cls_score,
            label,
            weight,
            class_weight=class_weight,
            reduction=reduction,
            avg_factor=avg_factor,
            avg_non_ignore=self.avg_non_ignore,
            ignore_index=self.ignore_index,
            **kwargs)
        return loss_cls

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

