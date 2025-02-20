# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    # DACS Self-Training with SegFormer Network Architecture
    '../_base_/uda/uda_segformer_mit-b0_dacs.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_oem2oem_768x768_strong-augs-src.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/schedule_80k.py',
    '../_base_/default_runtime.py',
]
crop_size = (768, 768)
stride = (384, 384)
data_preprocessor = dict(
    size=crop_size,
    seg_pad_val=0
)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa
model = dict(
    debug_img_interval=500,
    pseudo_threshold=0.968,
    debug=True,
    segmentor=dict(
        backbone=dict(
            init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
            embed_dims=64,
            num_heads=[1, 2, 5, 8],
            num_layers=[3, 6, 40, 3]
        ),
        decode_head=dict(
            in_channels=[64, 128, 320, 512],
            num_classes=9,
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, ignore_index=255, reduction='none'),
                dict(type='LovaszLoss', loss_name='loss_lovasz', per_image=True, loss_weight=1.0, reduction='none')
            ]
        ),
        train_cfg=dict(
            work_dir='work_dirs/02_21_uda_segformer_mit-b5_4xb2-80k_oem-768x768-alld_ignore255_dacs_ce-lovasz_th0.968'
        ),
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride)
    )
)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
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

train_dataloader = dict(batch_size=2, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, max_keep_ckpts=2,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))
