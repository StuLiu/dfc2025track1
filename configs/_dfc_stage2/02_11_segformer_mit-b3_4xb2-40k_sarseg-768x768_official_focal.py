_base_ = [
    './02_08_segformer_mit-b3_4xb2-40k_sarseg-768x768_official.py'
]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='FocalLoss',
                alpha=1.0, gamma=2.0, reduction='mean', ignore_label=0
            )
        ]
    )
)