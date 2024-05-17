_base_ = ['./05_17_swin-tiny-patch4-window7-in1k-pre_deeplabv3plus_4xb4-80k_agrivision-512x512.py',]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth'  # noqa
model = dict(
    backbone=dict(
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    decode_head=dict(
        # in_channels=[192, 384, 768, 1536]
        in_channels=1536,
        channels=768,
        c1_in_channels=192,
        c1_channels=144,
    ),
    auxiliary_head=dict(
        in_channels=768,
        channels=512
    )
)