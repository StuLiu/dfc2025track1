_base_ = [
    '../_base_/models/deeplabv3plus_swin.py', '../_base_/datasets/agriculture_vision.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(
    mean=[111.46, 113.90, 112.23, 118.30],
    std=[43.75, 41.29, 41.72, 46.56],
    bgr_to_rgb=False,
    size=crop_size
)
data_root = '/home/liuwang/liuwang_data/documents/datasets/seg/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN'
train_pipeline = [
    dict(type='LoadTifImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(512, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal']),
    dict(type='RandomFlip', prob=0.5, direction=['vertical']),
    dict(type='RandomRotate90', prob=0.5, degree=90),
    dict(type='PhotoMetricDistortionTif'),
    dict(type='PackSegInputs')
]
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        pipeline=train_pipeline,
        data_prefix=dict(
            img_path='img_dir/train_val_mosaic',
            seg_map_path='ann_dir/train_val_mosaic'),
    ),
    batch_size=4,
    num_workers=4,
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
    ),
    batch_size=4,
    num_workers=4
)
test_dataloader = val_dataloader

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    # find_unused_parameters=False,
    data_preprocessor=data_preprocessor,
    backbone=dict(
        in_channels=4,
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        use_abs_pos_embed=False,
        drop_path_rate=0.3,
        patch_norm=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    decode_head=dict(
        # in_channels=[96, 192, 384, 768],
        in_channels=768,
        c1_in_channels=96,
        c1_channels=72,
        num_classes=9,
        loss_decode=dict(type='ACWLoss')
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=9,
        loss_decode=dict(type='ACWLoss')
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=8000,
        max_keep_ckpts=2,
        save_last=True,
        save_best=['mIoU'],
        type='CheckpointHook'
    )
)