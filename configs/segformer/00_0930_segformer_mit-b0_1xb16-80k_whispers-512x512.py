_base_ = [
    '../_base_/models/segformer_mit-b0.py', '../_base_/datasets/mmseg_yreb_512x512.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    seg_pad_val=255.,
    mean = [1368.611, 1159.376, 1066.220, 1000.479, 1233.831,
            1868.624, 2124.804, 2111.797, 2322.861, 1078.598, 1715.745, 1081.012,
            -1487.655, -803.544],
    std = [491.791, 544.063, 557.462, 675.875, 660.595,
           602.205, 635.313, 645.017, 678.677, 557.957, 665.963, 528.291,
           430.152, 362.631],
    bgr_to_rgb = False,
)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        in_channels=14,
    ),
    decode_head=dict(num_classes=10))

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

# train_dataloader = dict(batch_size=4, num_workers=4)
# val_dataloader = dict(batch_size=1, num_workers=4)
# test_dataloader = dict(batch_size=1, num_workers=4)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, max_keep_ckpts=2,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))
