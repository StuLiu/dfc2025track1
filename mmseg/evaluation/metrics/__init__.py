# Copyright (c) OpenMMLab. All rights reserved.
from .citys_metric import CityscapesMetric
from .depth_metric import DepthMetric
from .iou_metric import IoUMetric
# custom
from .iou_metric import MultiLabelIoUMetric


__all__ = ['IoUMetric', 'CityscapesMetric', 'DepthMetric',
           # custom
           'MultiLabelIoUMetric',
    ]
