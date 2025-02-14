_base_ = [
    './02_12_segformer_mit-b3_4xb2-80k_oem-768x768-alld.py',
]

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b5_20220624-658746d9.pth'  # noqa

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 6, 40, 3]
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512]
    )
)


train_cfg = dict(type='IterBasedTrainLoop', max_iters=80000, val_interval=8000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=8000, max_keep_ckpts=2,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))