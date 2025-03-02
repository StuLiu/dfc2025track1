# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    # DACS Self-Training with SegFormer Network Architecture
    '../_base_/uda/uda_upernet_swinv2-base_dacs.py',
    # GTA->Cityscapes Data Loading
    '../_base_/datasets/uda_oem2oem_1024x1024_saugs_downup.py',
    # Linear Learning Rate Warmup with Subsequent Linear Decay
    '../_base_/schedules/schedule_80k.py',
    '../_base_/default_runtime.py',
]
crop_size = (1024, 1024)
stride = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    seg_pad_val=0
)
model = dict(
    type='DACSV2',
    debug_img_interval=500,
    debug=True,
    segmentor=dict(
        decode_head=dict(
            num_classes=9,
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, ignore_index=255, reduction='none')
            ]
        ),
        auxiliary_head=dict(
            num_classes=9,
            loss_decode=[
                dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, ignore_index=255, reduction='none')
            ]
        ),
        train_cfg=dict(
            work_dir='work_dirs/03_02_uda_upernet_swinv2-base_4xb1-80k_oem-1024x1024-alld_ignore255_dacsv2_ce_th0.968_downup'
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

train_dataloader = dict(batch_size=1, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, max_keep_ckpts=2,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))
