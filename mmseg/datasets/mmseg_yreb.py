# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MMSegYREBDataset(BaseSegDataset):
    """LoveDA dataset.

    In segmentation map annotation for LoveDA, 0 is the ignore index.
    ``reduce_zero_label`` should be set to True. The ``img_suffix`` and
    ``seg_map_suffix`` are both fixed to '.png'.
    """

    METAINFO = dict(
        classes = ('Background', 'Tree', 'Grassland', 'Cropland', 'LowVegetation',
                   'Wetland', 'Water', 'BuiltUp', 'BareGround', 'Snow'),
        palette = [[0, 0, 0], [0, 255, 0], [128, 128, 0], [0, 128, 128], [0, 128, 0],
                   [0, 0, 128], [0, 0, 255], [128, 0, 0], [128, 128, 128], [255, 255, 255]]
    )

    def __init__(self,
                 img_suffix='.tif',
                 seg_map_suffix='.tif',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
