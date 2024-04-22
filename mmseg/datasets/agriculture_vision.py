from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class AgricultureVisionDataset(BaseSegDataset):

    METAINFO = dict(
        classes=(
            "background",
            "double_plant",
            "drydown",
            "endrow",
            "nutrient_deficiency",
            "planter_skip",
            "water",
            "waterway",
            "weed_cluster",
        ),
        palette=[
            [0, 0, 0],
            [0, 0, 63],
            [0, 63, 63],
            [0, 63, 0],
            [0, 63, 127],
            [0, 63, 191],
            [0, 63, 255],
            [0, 127, 63],
            [0, 127, 127],
        ],
    )

    def __init__(
        self,
        img_suffix=".tif",
        seg_map_suffix=".png",
        ignore_index=255,
        **kwargs
    ) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            **kwargs
        )
