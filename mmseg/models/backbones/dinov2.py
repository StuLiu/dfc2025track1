"""
@Project : mmseg-agri
@File    : dinov2.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/2/12 下午1:26
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn

from mmengine.model import BaseModule
from mmseg.registry import MODELS
import torch.distributed as dist


@MODELS.register_module()
class DinoV2(BaseModule):
    def __init__(
        self,
        in_channels=3,
        img_size: int = 224,
        backbone_name: str = "dinov2_vitb14",
        pretrained: str = None,
        **kwargs,
    ):
        super().__init__()
        assert in_channels == 3, 'in_channels must equal to 3'
        self.in_channels = in_channels

        self.patch_size = 14
        self.img_size = img_size
        assert img_size % self.patch_size == 0, f'img_size {img_size} % {self.patch_size} != 0'

        # Define the backbone networks (a vision transformer)
        # note: The pretrained weight will be loaded into this model
        self.encoder = torch.hub.load('dinov2', backbone_name, source='local')

        self.encoder_depth = len(self.encoder.blocks)

        # if pretrained is not None:
        #     self.init_pretrained(pretrained, True)

    def freeze_backbone(self, freezing=True):
        for param in self.encoder.parameters():
            param.requires_grad = (not freezing)

    @staticmethod
    def token_to_image(x, shape, remove_class_token=True):
        if remove_class_token:
            x = x[:, 1:]
        x = x.permute(0, 2, 1)
        x = x.view(shape).contiguous()
        return x

    def forward_features(self, img: torch.Tensor):
        b, c, h, w = img.shape
        token_img_shape = (b, self.encoder.embed_dim, h // self.patch_size, w // self.patch_size)

        x_patch = self.encoder.patch_embed(img)

        x = torch.cat((self.encoder.cls_token.expand(x_patch.shape[0], -1, -1), x_patch), dim=1)
        x = x + self.encoder.interpolate_pos_encoding(x, w, h)
        x = x.contiguous()
        for i in range(self.encoder_depth):
            x = self.encoder.blocks[i](x)
        x = self.encoder.norm(x)
        x = self.token_to_image(x, token_img_shape)
        return x

    def forward(self, img):
        out = [self.forward_features(img)]
        return out

    def init_pretrained(self, pretrained: str = None, strict=False) -> None:
        if pretrained:
            state_dict = torch.load(pretrained, map_location='cpu')
            self.encoder.load_state_dict(state_dict, strict=strict)
            pass
