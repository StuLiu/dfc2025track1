"""
@Project : mmseg-agri
@File    : distributed.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2025/2/16 下午7:50
@e-mail  : 1183862787@qq.com
"""

# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, Union

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from mmengine.optim import OptimWrapper
from mmengine.registry import MODEL_WRAPPERS
from mmengine.model.utils import detect_anomalous_params
from mmengine.model.wrappers.distributed import MMDistributedDataParallel
# MODEL_WRAPPERS.register_module(module=DistributedDataParallel)
# MODEL_WRAPPERS.register_module(module=DataParallel)


@MODEL_WRAPPERS.register_module()
class UDA_MMDistributedDataParallel(MMDistributedDataParallel):
    def __init__(self,
                 module,
                 detect_anomalous_params: bool = False,
                 **kwargs):
        super().__init__(module=module, detect_anomalous_params=detect_anomalous_params, **kwargs)

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Interface for model forward, backward and parameters updating during
        training process.

        :meth:`train_step` will perform the following steps in order:

        - If :attr:`module` defines the preprocess method,
          call ``module.preprocess`` to pre-processing data.
        - Call ``module.forward(**data)`` and get losses.
        - Parse losses.
        - Call ``optim_wrapper.optimizer_step`` to update parameters.
        - Return log messages of losses.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): A wrapper of optimizer to
                update parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # call UDA_Decorator.train_step(data, optim_wrapper) -> dict
        log_vars = self.module.train_step(data, optim_wrapper)
        return log_vars