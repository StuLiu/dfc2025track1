# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from timm.layers import LayerNorm2d

from mmseg.registry import MODELS
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class VitSegHead(BaseDecodeHead):
    """DeConvolution Networks for Semantic Segmentation.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(self.in_channels,
                               self.in_channels // 2,
                               kernel_size=2, stride=2, padding=0, bias=False
                               ),
            LayerNorm2d(self.in_channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(self.in_channels // 2,
                               self.in_channels // 4,
                               kernel_size=2, stride=2, padding=0, bias=False
                               ),
            LayerNorm2d(self.in_channels // 4),
            nn.Conv2d(
                self.in_channels // 4,
                self.in_channels // 4,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(self.in_channels // 4),
        )
        self.conv_seg = nn.Conv2d(self.in_channels // 4, self.out_channels, kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        output = self.upscale(inputs[-1])
        output = self.cls_seg(output)
        return output
