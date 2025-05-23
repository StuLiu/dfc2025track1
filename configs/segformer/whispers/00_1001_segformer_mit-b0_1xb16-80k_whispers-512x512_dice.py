_base_ = ['./00_0930_segformer_mit-b0_1xb16-80k_whispers-512x512.py']

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
            dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
        ]
    )
)