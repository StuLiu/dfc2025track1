_base_ = [
    './02_09_segformer_mit-b5_4xb2-160k_sarseg-768x768_interv1_ignore255_sce-dice.py',
]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='GeneralizedCELoss',
                q=0.5,
                loss_name='loss_gce',
                loss_weight=1.0,
                ignore_index=255
            ),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=3.0, ignore_index=255)
        ]
    ),
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=160000, val_interval=16000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=16000, max_keep_ckpts=2,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))
