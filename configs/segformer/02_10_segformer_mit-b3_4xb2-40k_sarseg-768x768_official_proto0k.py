_base_ = [
    '02_08_segformer_mit-b3_4xb2-40k_sarseg-768x768_official.py',
]

model = dict(
    type='EncoderDecoderPrototypeLearning',
    prototype_cfg=dict(
        num_classes=9,
        feat_channels=512,
        ignore_index=0,
        momentum=0.998,
        resume='',
        debug=True
    ),
    warmup_iters=0,
)

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=4000, max_keep_ckpts=2,
                    save_last=True, save_best=['mIoU'], type='CheckpointHook'))