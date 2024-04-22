_base_ = ['./segformer_mit-b0_8xb2-160k_agriculture-vision-512x512.py']

checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'  # noqa

data_preprocessor = dict(
    mean=[111.46, 113.90, 112.23],
    std=[43.75, 41.29, 41.72],
)

# model settings
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3]),
    decode_head=dict(in_channels=[64, 128, 320, 512]))

train_dataloader = dict(batch_size=8, num_workers=4)
