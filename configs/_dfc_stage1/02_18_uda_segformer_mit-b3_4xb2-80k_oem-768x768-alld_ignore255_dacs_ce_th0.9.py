# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

_base_ = [
    './02_17_uda_segformer_mit-b3_4xb2-40k_oem-768x768-alld_ignore255_ce.py',
]

model=dict(
    pseudo_threshold=0.9,
    segmentor=dict(
        train_cfg=dict(
            work_dir='work_dirs/02_18_uda_segformer_mit-b3_4xb2-80k_oem-768x768-alld_ignore255_dacs_ce_th0.9'
        )
    )
)

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

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=4000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, max_keep_ckpts=2,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))
