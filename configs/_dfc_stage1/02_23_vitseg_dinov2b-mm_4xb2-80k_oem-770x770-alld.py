_base_ = [
    '../_base_/models/vitseg_dinov2-b-mm.py',
    '../_base_/datasets/open_earth_map_768x768.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (770, 770)
data_preprocessor = dict(
    size=crop_size,
    seg_pad_val=0,
)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        img_size=crop_size,
    ),
    decode_head=dict(
        num_classes=9,
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, ignore_index=255)
        ]
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(770, 770), stride=(385, 385))
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

train_dataloader = dict(batch_size=2, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=2)
test_dataloader = dict(batch_size=1, num_workers=2)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, max_keep_ckpts=1,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))
