# Copyright (c) OpenMMLab. All rights reserved.
from .accuracy import Accuracy, accuracy
from .boundary_loss import BoundaryLoss
from .cross_entropy_loss import (CrossEntropyLoss, binary_cross_entropy,
                                 cross_entropy, mask_cross_entropy)
from .dice_loss import DiceLoss
# from .focal_loss import FocalLoss
from .huasdorff_distance_loss import HuasdorffDisstanceLoss
from .lovasz_loss import LovaszLoss
from .ohem_cross_entropy_loss import OhemCrossEntropy
from .silog_loss import SiLogLoss
from .tversky_loss import TverskyLoss
from .utils import reduce_loss, weight_reduce_loss, weighted_loss
# custom
from .cross_entropy_loss import (ACWLoss, ACWJaccardLoss, ACWLossV2, ACWFocalLoss, ACWDefocalLoss, ACWSCELoss,
                                 HybridV1, HybridV2, SymmetricCELoss, GeneralizedCELoss,
                                 ReversedFocalLoss, FocalLoss)
from .multi_label_acw_loss import MultiLabelBCELoss, MultiLabelACWLoss, MultiLabelJaccardLoss


__all__ = [
    'accuracy', 'Accuracy', 'cross_entropy', 'binary_cross_entropy',
    'mask_cross_entropy', 'CrossEntropyLoss', 'reduce_loss',
    'weight_reduce_loss', 'weighted_loss', 'LovaszLoss', 'DiceLoss',
    'FocalLoss', 'TverskyLoss', 'OhemCrossEntropy', 'BoundaryLoss',
    'HuasdorffDisstanceLoss', 'SiLogLoss',
    # custom
    'ACWLoss', 'ACWJaccardLoss', 'ACWLossV2', 'ACWDefocalLoss', 'ACWFocalLoss', 'ACWSCELoss',
    'MultiLabelBCELoss', 'MultiLabelACWLoss', 'MultiLabelJaccardLoss',
    'SymmetricCELoss', 'GeneralizedCELoss', 'ReversedFocalLoss',
    "HybridV1",
    "HybridV2",
]
