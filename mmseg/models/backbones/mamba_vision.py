# Copyright (c) OpenMMLab. All rights reserved.

import torch
import mambavision

from mmengine.model import BaseModule
from mmengine.registry import MODELS as MMENGINE_MODELS

from mmseg.registry import MODELS


# @MODELS.register_module()
class MMSegMambaVision(BaseModule):
    """Wrapper to use backbones from timm library. More details can be found in
    `NVlabs <https://github.com/NVlabs/MambaVision/tree/main>`_ .

    Args:
        model_name (str): Name of timm model to instantiate.
        pretrained (bool): Load pretrained weights if True.
        checkpoint_path (str): Path of checkpoint to load after
            model is initialized.
        in_channels (int): Number of input image channels. Default: 3.
        init_cfg (dict, optional): Initialization config dict
        **kwargs: Other timm & model specific arguments.
    """

    def __init__(
        self,
        model_name='mamba_vision_B',
        pretrained=False,
        checkpoint_path='/mnt/home/liuwang_data/documents/pretrained/backbones/mambavision/mambavision_base_1k.pth.tar',
        in_channels=3,
        init_cfg=None,
        **kwargs,
    ):
        if mambavision is None:
            raise RuntimeError('timm is not installed')
        assert model_name in [
            'mamba_vision_T',
            'mamba_vision_T2',
            'mamba_vision_S',
            'mamba_vision_B',
            'mamba_vision_L',
            'mamba_vision_L2'
        ]
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.model = mambavision.create_model(
            model_name=model_name,
            pretrained=pretrained,
            model_path=checkpoint_path,
            **kwargs
        )
        self.model.head = None

        # Hack to use pretrained weights from timm
        if pretrained or checkpoint_path:
            self._is_init = True

    def forward(self, x):
        features = self.model.forward_features(x)
        return features


if __name__ == '__main__':
    import torch

    x = torch.rand(1, 3, 512, 512).cuda()  # place image on cuda
    mambavision_model = MMSegMambaVision(
        model_name='mamba_vision_B',
        pretrained=False,
        checkpoint_path='/mnt/home/liuwang_data/documents/pretrained/backbones/mambavision/mambavision_base_1k.pth.tar',
        in_channels=3,
        init_cfg=None
    ).cuda()
    y = mambavision_model(x)
    print(y.shape)


