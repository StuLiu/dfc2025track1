_base_ = ['./00_0930_segformer_mit-b0_1xb16-80k_whispers-512x512.py']

crop_size = (512, 512)
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(type='LoadTifImageFromFileV2'),
    dict(type='LoadAnnotations'),
    dict(type='RandomFlip', prob=[0.5, 0.5], direction=['horizontal', 'vertical']),
    dict(type='RandomRotateRectangle', prob=1.0),
    dict(type='PhotoMetricDistortionTifWhispers'),
    dict(
        type='RandomResize',
        scale=(256, 256),
        ratio_range=(2.0, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)