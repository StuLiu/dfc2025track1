# dataset settings
dataset_type = 'MMSegYREBDataset'
data_root = '/home/liuwang/liuwang_data/documents/datasets/seg/challenges/MMSeg-YREB'
crop_size = (256, 256)
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadTifImageFromFileV2'),
    dict(type='LoadAnnotations'),
    # dict(
    #     type='RandomResize',
    #     scale=(256, 256),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(type='RandomRotateRectangle', prob=1.0),
    # dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadTifImageFromFileV2'),
    dict(type='Resize', scale=(256, 256), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadTifImageFromFileV2'),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='train/merge', seg_map_path='train/label'),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='val/merge', seg_map_path='val/label'),
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='test/merge', seg_map_path='test/label_zeros'),
        pipeline=test_pipeline))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
