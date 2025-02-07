_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/dfc2025sarseg.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
crop_size = (768, 768)
data_preprocessor = dict(
    size=crop_size,
    seg_pad_val=0,
)

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3]
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        num_classes=9,
        loss_decode=[
            dict(type='SymmetricCELoss', loss_name='loss_ce', loss_weight=1.0, ignore_index=0)
        ]
    ),
    test_cfg=dict(mode='slide', crop_size=(768, 768), stride=(384, 384))
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
        end=40000,
        by_epoch=False,
    )
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    dataset = dict(
        data_prefix=dict(
            seg_map_path='train/labels_pl/labels_intersection'
        )
    )
)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, max_keep_ckpts=1,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))