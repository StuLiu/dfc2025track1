_base_ = [
    './02_08_segformer_mit-b3_4xb2-40k_sarseg-768x768_official.py'
]

model = dict(
    decode_head=dict(
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                loss_name='loss_ce',
                loss_weight=1.0,
                ignore_index=255
            ),
            dict(
                type='LovaszLoss',
                classes='present',
                per_image=False,
                reduction='none',
                loss_weight=1.0,
                loss_name='loss_lovasz'
            )
        ]
    )
)