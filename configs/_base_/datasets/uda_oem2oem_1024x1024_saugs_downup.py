# dataset settings
dataset_type = 'OpenEarthMapDataset'
data_root = '/home/liuwang/liuwang_data/documents/datasets/seg/OpenEarthMap_mmseg'
crop_size = (1024, 1024)
train_pipeline_src = [
    dict(type='LoadTifImageFromFileV2', to_float32=False),
    dict(type='LoadTifAnnotationsV2'),
    dict(
        type='Pad',
        size=(1024, 1024),
        pad_val=dict(img=0, seg=0)
    ),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75, ignore_index=255),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='RandomRotate90', degree=90, prob=0.5),
    dict(type='RandomDownUp', scale=crop_size, ratio_range=(0.1, 0.35), keep_ratio=True, up_scale_max=3, prob=0.2),
    dict(type='RandomCrop', crop_size=crop_size, ignore_index=255),
    dict(type='PhotoMetricDistortionV2'),
    dict(
        type='Pad',
        size=crop_size,
        pad_val=dict(img=0, seg=0)
    ),
    dict(type='PackSegInputs')
]
train_pipeline_tgt = [
    dict(type='LoadTifImageFromFileV2', to_float32=False),
    dict(type='LoadTifAnnotationsV2'),
    dict(
        type='Pad',
        size=(1024, 1024),
        pad_val=dict(img=0, seg=0)
    ),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75, ignore_index=255),
    dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical']),
    dict(type='RandomRotate90', degree=90, prob=0.5),
    dict(
        type='Pad',
        size=crop_size,
        pad_val=dict(img=0, seg=0)
    ),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadTifImageFromFileV2', to_float32=False),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadTifAnnotationsV2'),
    dict(type='PackSegInputs')
]
img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadTifImageFromFileV2', to_float32=False),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='vertical'),
                dict(type='RandomFlip', prob=1., direction=['horizontal', 'vertical'])
            ], [dict(type='LoadTifAnnotationsV2')], [dict(type='PackSegInputs')]
        ])
]
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='UDADataset',
        source=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='images/train_val', seg_map_path='annotations/train_val'),
            pipeline=train_pipeline_src
        ),
        target=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                # img_path='images/test', seg_map_path='annotations/test'),
                img_path='images/test_with_sarseg', seg_map_path='annotations/test_with_sarseg'),
            pipeline=train_pipeline_tgt
        ),
        rare_class_sampling=None
        # rare_class_sampling=dict(
        #     min_pixels=3000,
        #     class_temp=0.01,
        #     min_crop_ratio=0.5
        # )
    ),

)
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val', seg_map_path='annotations/val'),
        pipeline=test_pipeline
    )
)
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test', seg_map_path='annotations/test'),
        pipeline=test_pipeline
    )
)

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator
