_base_ = [
    './02_08_segformer_mit-b3_4xb2-40k_sarseg-768x768_official.py'
]

train_dataloader = dict(
    dataset=dict(
        data_prefix=dict(
            img_path='train/sar_images_lee_k19'
        )
    )
)
val_dataloader = dict(
    dataset = dict(
        data_prefix=dict(
            img_path='val/sar_images_lee_k19'
        )
    )
)
test_dataloader = val_dataloader
