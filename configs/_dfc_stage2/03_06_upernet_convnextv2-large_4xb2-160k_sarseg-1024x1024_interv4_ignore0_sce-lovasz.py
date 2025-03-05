_base_ = [
    '../_base_/models/upernet_convnextv2-large.py',
    '../_base_/datasets/dfc2025sarseg1024x1024.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(
    size=crop_size,
    seg_pad_val=0,
)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=9,
        ignore_index=0,
        loss_decode=[
            dict(type='SymmetricCELoss', loss_name='loss_sce', loss_weight=1.0, ignore_index=0),
            dict(
                type='LovaszLoss',
                classes='present',
                per_image=False,
                reduction='none',
                loss_weight=1.0,
                loss_name='loss_lovasz'
            )
        ]
    ),
    auxiliary_head=dict(
        num_classes=9,
        ignore_index=0,
        loss_decode=[
            dict(type='SymmetricCELoss', loss_name='loss_sce', loss_weight=1.0, ignore_index=0),
            dict(
                type='LovaszLoss',
                classes='present',
                per_image=False,
                reduction='none',
                loss_weight=1.0,
                loss_name='loss_lovasz'
            )
        ]
    ),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(512, 512))
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0),
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
        end=160000,
        by_epoch=False,
    )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=16,
    dataset=dict(
        data_prefix=dict(
            seg_map_path='train/labels_pl/stage1_28_30_31_32_ensemble_best'
        )
    )
)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=16000, max_keep_ckpts=2,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))