# model settings
# custom_imports = dict(imports='mmpretrain.models', allow_failed_imports=False)
norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file=('https://download.openmmlab.com/mmclassification/v0/swin-v2/'
                 'swinv2-large-w24_in21k-pre_3rdparty_in1k-384px_20220803-3b36c165.pth')
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='mmpretrain.models.backbones.SwinTransformerV2',
        arch='large',
        img_size=384,
        patch_size=4,
        in_channels=3,
        window_size=[24, 24, 24, 12],
        drop_rate=0.,
        drop_path_rate=0.1,
        out_indices=[0, 1, 2, 3],
        use_abs_pos_embed=False,
        interpolate_mode='bicubic',
        with_cp=False,
        frozen_stages=-1,
        norm_eval=False,
        pad_small_map=False,
        norm_cfg=dict(type='LN'),
        stage_cfgs=dict(),
        patch_cfg=dict(),
        pretrained_window_sizes=[12, 12, 12, 6],
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            prefix='backbone.'
        ),
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[192, 384, 768, 1536],
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
        in_channels=768,
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
    test_cfg=dict(mode='slide', crop_size=(384, 384), stride=(192, 192)))
