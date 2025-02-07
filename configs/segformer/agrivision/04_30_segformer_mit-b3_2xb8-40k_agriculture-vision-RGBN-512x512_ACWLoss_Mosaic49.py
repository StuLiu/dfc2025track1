_base_ = ['./segformer_mit-b0_8xb2-160k_agriculture-vision-512x512.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa

data_preprocessor = dict(
    mean=[111.46, 113.90, 112.23, 118.30],
    std=[43.75, 41.29, 41.72, 46.56],
    bgr_to_rgb=False,
)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, max_keep_ckpts=4,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))

# data_root = '/mnt/home/wangzhiyu_data/Data/Challenge/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN/'
data_root = 'data/2024-CVPR-Agriculture-Vision/supervised/Agriculture-Vision-2021-MMSeg-RGBN'
train_pipeline = [
    dict(type='LoadTifImageFromFile'),
    dict(type='LoadAnnotations'),
    # dict(
    #     type='RandomResize',
    #     scale=(512, 512),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal']),
    dict(type='RandomFlip', prob=0.5, direction=['vertical']),
    dict(type='RandomRotate90', prob=0.5, degree=90),
    dict(type='PhotoMetricDistortionTif'),
    dict(type='PackSegInputs')
]
dataset_train_1 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(  # This is the original config of Dataset_A
        type='AgricultureVisionDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train', seg_map_path='ann_dir/train'),
        pipeline=train_pipeline,
    )
)
dataset_train_2 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(  # This is the original config of Dataset_A
        type='AgricultureVisionDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train_4mosaic', seg_map_path='ann_dir/train_4mosaic'),
        pipeline=train_pipeline,
    )
)
dataset_train_3 = dict(
    type='RepeatDataset',
    times=1,
    dataset=dict(  # This is the original config of Dataset_A
        type='AgricultureVisionDataset',
        data_root=data_root,
        data_prefix=dict(img_path='img_dir/train_9mosaic', seg_map_path='ann_dir/train_9mosaic'),
        pipeline=train_pipeline,
    )
)
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[dataset_train_1, dataset_train_2, dataset_train_3]
    ),
)
val_dataloader = dict(
    dataset=dict(
        data_root=data_root
    )
)
test_dataloader = val_dataloader
img_ratios = [0.3125, 0.5, 1.0]
tta_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadTifImageFromFile'),
    dict(
        type='TestTimeAug',
        transforms=[
            [dict(type='Resize', scale_factor=r, keep_ratio=True) for r in img_ratios],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
            ],
            [dict(type='LoadAnnotations')],
            [dict(type='PackSegInputs')]
        ]
    )
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
        loss_decode=dict(type='ACWLoss')))

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
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
