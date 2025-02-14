_base_ = [
    './02_12_segformer_mit-b3_4xb2-80k_oem-768x768-alld.py',
]
model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, ignore_index=255),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, ignore_index=255)
        ]
    ),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=20000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=20000, max_keep_ckpts=1,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))