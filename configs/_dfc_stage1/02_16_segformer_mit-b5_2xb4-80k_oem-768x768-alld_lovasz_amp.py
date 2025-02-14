_base_ = [
    './02_14_segformer_mit-b5_4xb2-80k_oem-768x768-alld.py',
]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0, ignore_index=255),
            dict(
                type='LovaszLoss',
                classes='present',
                per_image=False,
                reduction='none',
                loss_weight=1.0,
                loss_name='loss_lovasz'
            )
        ]
    )
)

train_dataloader = dict(batch_size=4, num_workers=8)
train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=40000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=20000, max_keep_ckpts=1,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))