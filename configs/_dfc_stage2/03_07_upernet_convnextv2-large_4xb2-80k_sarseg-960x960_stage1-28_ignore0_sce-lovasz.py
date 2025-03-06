_base_ = [
    '../_base_/models/upernet_convnextv2-large.py',
    '../_base_/datasets/dfc2025sarseg960x960.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_80k.py'
]
crop_size = (960, 960)
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
    test_cfg=dict(mode='slide', crop_size=(960, 960), stride=(480, 480))
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
        end=80000,
        by_epoch=False,
    )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=16,
    dataset=dict(
        data_prefix=dict(
            seg_map_path='train/labels_pl/02_26_uda_segformer_mit-b3_2xb4-80k_oem-768x768-alld_ignore255_dacsv2_ce_th0.968_downup_tta'
        )
    )
)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, max_keep_ckpts=2,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))