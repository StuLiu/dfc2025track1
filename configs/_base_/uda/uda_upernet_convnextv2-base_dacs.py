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
checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-379425cc.pth'
model = dict(
    type='DACS',
    segmentor=dict(
        type='EncoderDecoder',
        data_preprocessor=data_preprocessor,
        pretrained=None,
        backbone=dict(
            type='mmpretrain.ConvNeXt',
            arch='base',
            in_channels=3,
            use_grn=True,
            drop_path_rate=0.0,
            layer_scale_init_value=0.,
            stem_patch_size=4,
            norm_cfg=dict(type='LN2d', eps=1e-6),
            act_cfg=dict(type='GELU'),
            linear_pw_conv=True,
            frozen_stages=0,
            gap_before_final_norm=False,
            with_cp=False,
            out_indices=[0, 1, 2, 3],
            init_cfg=dict(
                type='Pretrained',
                checkpoint=checkpoint_file,
                prefix='backbone.'
            )
        ),
        decode_head=dict(
            type='UPerHead',
            in_channels=[128, 256, 512, 1024],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=512,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        auxiliary_head=dict(
            type='FCNHead',
            in_channels=512,
            in_index=2,
            channels=256,
            num_convs=1,
            concat_input=False,
            dropout_ratio=0.1,
            num_classes=19,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
        # model training and testing settings
        train_cfg=dict(),
        test_cfg=dict(mode='whole'),
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
