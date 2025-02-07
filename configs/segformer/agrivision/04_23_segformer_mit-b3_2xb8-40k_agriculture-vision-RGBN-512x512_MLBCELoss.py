_base_ = ['./segformer_mit-b0_8xb2-160k_agriculture-vision-512x512.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa
crop_size = (512, 512)
data_preprocessor = dict(
    mean=[111.46, 113.90, 112.23, 118.30],
    std=[43.75, 41.29, 41.72, 46.56],
    bgr_to_rgb=False,
    seg_pad_val=0,
    size=crop_size,
)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, max_keep_ckpts=4,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))
data_root = 'data/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN-MultiLabel'
train_pipeline = [
    dict(type='LoadTifImageFromFile'),
    dict(type='LoadTifAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal']),
    dict(type='RandomFlip', prob=0.5, direction=['vertical']),
    dict(type='RandomRotate90', prob=0.5, degree=90),
    dict(type='PhotoMetricDistortionTif'),
    dict(type='MultiLabelPackSegInputs')
]
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type='MultiLabelAgricultureVisionDataset',
        data_root=data_root,
        pipeline=train_pipeline,
    )
)
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadTifImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadTifAnnotations'),
    dict(type='MultiLabelPackSegInputs')
]
val_dataloader = dict(
    dataset=dict(
        type='MultiLabelAgricultureVisionDataset',
        data_root=data_root,
        pipeline=test_pipeline,
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(type='MultiLabelIoUMetric')
test_evaluator = dict(type='MultiLabelIoUMetric')

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadTifImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ],
            [dict(type='LoadTifAnnotations')],
            [dict(type='MultiLabelPackSegInputs')]
        ])
]

# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=4,
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        loss_decode=dict(type='MultiLabelBCELoss')))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False,
    )
]
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)  # #####
