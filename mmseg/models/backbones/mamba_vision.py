# Copyright (c) OpenMMLab. All rights reserved.

import torch

from transformers import AutoModel
from timm.data.transforms_factory import create_transform

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
        model_name='MambaVision-B-1K',
        checkpoint_path=None,
        in_channels=3,
        init_cfg=None,
        **kwargs,
    ):
        assert model_name in [
            'MambaVision-T-1K', 'MambaVision-T2-1K', 'MambaVision-S-1K',
            'MambaVision-B-1K', 'MambaVision-L-1K', 'MambaVision-L2-1K'
        ]
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.model = AutoModel.from_pretrained(f"nvidia/{model_name}", trust_remote_code=True)
        self._is_init = True

    def forward(self, x):
        _, features = self.model(x)
        return features


if __name__ == '__main__':
    import torch

    x = torch.rand(1, 3, 512, 512).cuda()  # place image on cuda
    mambavision_model = MMSegMambaVision().cuda()
    y = mambavision_model(x)
    for y_ in y:
        print(y_.shape)


