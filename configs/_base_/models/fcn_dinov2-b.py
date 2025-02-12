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
        type='DinoV2',
        in_channels=3,
        img_size=672,
        backbone_name="dinov2_vitb14",
        pretrained='/home/liuwang/liuwang_data/documents/pretrained/backbones/vit/dinov2_vitb14_pretrain.pth',
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
