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


def convert_to_syncbn(module):
    """
    将模型中的 nn.BatchNorm 替换为 nn.SyncBatchNorm
    """
    # 遍历模型的所有 children
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm1d):
            setattr(module, name, nn.SyncBatchNorm(child.num_features))
        elif isinstance(child, nn.BatchNorm2d):
            setattr(module, name, nn.SyncBatchNorm(child.num_features))
        elif isinstance(child, nn.BatchNorm3d):
            setattr(module, name, nn.SyncBatchNorm(child.num_features))
        else:
            # 如果是其他层，递归替换其子层
            convert_to_syncbn(child)
    return module


vitseg_settings = {
    'dinov2_vits14': {
        'feat_channel': None
    },
    'dinov2_vitb14': {
        'feat_channel': 768
    },
    'dinov2_vitl14': {
        'feat_channel': 1024
    },
    'dinov2_vitg4': {
        'feat_channel': None
    }
}

@MODELS.register_module()
class DinoV2(BaseModule):
    def __init__(
        self,
        in_channels=3,
        img_size: int=224,
        backbone_name: str = "dinov2_vitb14",
        pretrained: str=None,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = 14

        assert in_channels == 3, 'in_channels must equal to 3}'
        self.in_channels = in_channels

        self.img_size = img_size

        assert img_size % 14 == 0, f'img_size {img_size} % 14 != 0'

        # self.encoder = torch.hub.load('facebookresearch/dinov2', model_name)
        self.encoder = torch.hub.load('dinov2', backbone_name, source='local')

        if pretrained is not None:
            self.init_pretrained(pretrained, True)

        # self.upscale = nn.Sequential(
        #     nn.ConvTranspose2d(self.encoder.embed_dim,
        #                        self.encoder.embed_dim // 2,
        #                        kernel_size=2, stride=2, padding=0, bias=False
        #                        ),
        #     LayerNorm2d(self.encoder.embed_dim // 2),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(self.encoder.embed_dim // 2,
        #                        self.encoder.embed_dim // 4,
        #                        kernel_size=2, stride=2, padding=0, bias=False
        #                        ),
        #     LayerNorm2d(self.encoder.embed_dim // 4),
        #     nn.Conv2d(
        #         self.encoder.embed_dim // 4,
        #         self.encoder.embed_dim // 4,
        #         kernel_size=3,
        #         padding=1,
        #         bias=False,
        #     ),
        #     LayerNorm2d(self.encoder.embed_dim // 4),
        # )
        # # self.token_masking = TokenMasking(self.encoder.mask_token)
        #
        # out = nn.Conv2d(self.encoder.embed_dim // 4, num_classes, kernel_size=1, padding=0, bias=False)
        # torch.nn.init.normal_(out.weight, 0, std=0.1)
        # self.out = out
        #
        # self.param_defs_decoder = [
        #     ("out", self.out),
        #     ("upscale", self.upscale)
        # ]
        #
        # self.param_defs_encoder_blocks = [
        #     ("encoder.blocks", self.encoder.blocks),
        # ]
        #
        # self.param_defs_encoder_stems = [
        #     ("encoder.mask_token", self.encoder.mask_token),
        #     ("encoder.norm", self.encoder.norm),
        #     ("encoder.pos_embed", self.encoder.pos_embed),
        #     ("encoder.patch_embed.proj", self.encoder.patch_embed.proj),
        #     ("encoder.cls_token", self.encoder.cls_token)
        #     if hasattr(self.encoder, "cls_token")
        #     else (None, None),
        # ]

        self.encoder_depth = len(self.encoder.blocks)

        # if dist.get_world_size() > 1:
        self.encoder = convert_to_syncbn(self.encoder)

    def token_to_image(self, x, shape, remove_class_token=True):
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
