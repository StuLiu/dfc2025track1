_base_ = ['./00_0930_segformer_mit-b0_1xb16-80k_whispers-512x512.py']

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='HybridV2', loss_weight=1.0),
        ]
    )
)