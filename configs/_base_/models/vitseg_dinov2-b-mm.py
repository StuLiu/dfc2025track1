# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.models.VisionTransformer',
        arch='base',
        img_size=224,
        patch_size=14,
        in_channels=3,
        out_indices=-1,
        drop_path_rate=0.1,
        qkv_bias=True,
        norm_cfg=dict(type='LN', eps=1e-6),
        final_norm=True,
        out_type='featmap',
        with_cls_token=True,
        frozen_stages=-1,
        interpolate_mode='bicubic',
        layer_scale_init_value=1e-5,
        patch_cfg=dict(),
        layer_cfgs=dict(),
        pre_norm=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint="https://download.openmmlab.com/mmpretrain/v1.0/dinov2/"
                       "vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth",
            prefix='backbone.'
        )
    ),
    decode_head=dict(
        type='VitSegHead',
        in_channels=768,
        channels=512,
        dropout_ratio=0.1,
        num_classes=9,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(672, 672), stride=(336, 336))
)
