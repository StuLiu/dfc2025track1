_base_ = [
    './02_08_segformer_mit-b3_4xb2-40k_sarseg-768x768_official.py'
]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='GeneralizedCELoss',
                q=0.5,
                loss_name='loss_gce',
                loss_weight=1.0,
                ignore_index=0
            )
        ]
    )
)