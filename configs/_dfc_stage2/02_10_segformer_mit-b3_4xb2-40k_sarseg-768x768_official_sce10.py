_base_ = [
    './02_08_segformer_mit-b3_4xb2-40k_sarseg-768x768_official.py'
]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(type='SymmetricCELoss', loss_name='loss_ce', loss_weight=1.0, ignore_index=0, beta=10)
        ]
    )
)
