_base_ = ['./05_17_swin-tiny-patch4-window7-in1k-pre_deeplabv3plus_4xb4-80k_agrivision-512x512.py',]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window7_224_20220317-e9b98025.pth'  # noqa
model = dict(
    backbone=dict(
        embed_dims=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)
    ),
    decode_head=dict(
        # in_channels=[128, 256, 512, 1024]
        in_channels=1024,
        c1_in_channels=128,
        c1_channels=96,
    ),
    auxiliary_head=dict(
        in_channels=512,
        channels=384
    )
)