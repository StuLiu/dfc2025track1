"""
@Project : mmseg-agri
@File    : uda_segformer_mit-b0_dacs.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/2/15 下午1:30
@e-mail  : 1183862787@qq.com
"""
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255
)
# noqa
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
model = dict(
    type='DACS',
    segmentor=dict(
        type='EncoderDecoder',
        backbone=dict(
            type='MixVisionTransformer',
            in_channels=3,
            embed_dims=32,
            num_stages=4,
            num_layers=[2, 2, 2, 2],
            num_heads=[1, 2, 5, 8],
            patch_sizes=[7, 3, 3, 3],
            sr_ratios=[8, 4, 2, 1],
            out_indices=(0, 1, 2, 3),
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint)
        ),
        decode_head=dict(
            type='SegformerHead',
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            channels=256,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
        ),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole')
    ),
    alpha=0.999,
    pseudo_threshold=0.968,
    ignore_index=255,
    pseudo_weight_ignore_margin=(0, 0, 0, 0),
    imnet_feature_dist_lambda=0,
    imnet_feature_dist_classes=None,
    imnet_feature_dist_scale_min_ratio=None,
    mix='class',
    blur=True,
    color_jitter_strength=0.2,
    color_jitter_probability=0.5,
    debug_img_interval=1000,
    palette='oem',  # default is open earth map
    print_grad_magnitude=False,
)
find_unused_parameters = True
use_ddp_wrapper = True
model_wrapper_cfg = dict(type='UDA_MMDistributedDataParallel')
