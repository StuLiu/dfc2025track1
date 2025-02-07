# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class DFC2025SarSegDataset(BaseSegDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """
    METAINFO = dict(
        classes=('background', 'Bareland', 'Rangeland', 'Developed_space', 'Road',
               'Tree', 'Water', 'Agriculture land', 'Building'),
        palette=[
            [0, 0, 0], [128, 0, 0], [0, 255, 32], [148, 148, 148], [255, 255, 255],
            [28, 97, 33], [0, 67, 255], [74, 182, 71], [223, 25, 1]
        ]
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix, seg_map_suffix=seg_map_suffix, **kwargs)
