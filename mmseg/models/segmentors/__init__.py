# Copyright (c) OpenMMLab. All rights reserved.
from .base import BaseSegmentor
from .cascade_encoder_decoder import CascadeEncoderDecoder
from .depth_estimator import DepthEstimator
from .encoder_decoder import EncoderDecoder
from .encoder_decoder_prototype_learning import EncoderDecoderPrototypeLearning
from .multimodal_encoder_decoder import MultimodalEncoderDecoder
from .seg_tta import SegTTAModel, SegTTAModelV2
from .dacs import DACS
from .dacsv2 import DACSV2
from .cutmix import CutMix


__all__ = [
    'BaseSegmentor', 'EncoderDecoder', 'CascadeEncoderDecoder', 'SegTTAModel',
    'MultimodalEncoderDecoder', 'DepthEstimator',
    # ###########################
    'EncoderDecoderPrototypeLearning', 'DACS', 'CutMix', 'DACSV2', 'SegTTAModelV2'
]
