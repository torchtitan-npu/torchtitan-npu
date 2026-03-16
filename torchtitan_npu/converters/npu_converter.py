# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Set, Type, TYPE_CHECKING, Union

import torch.nn as nn

from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .base_converter import BaseConverter


class NPUConverter(ModelConverter):

    _patch_cls: Type["BaseConverter"] = None
    _patch_name: str = None
    _supported_models: Set[str] = None

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        self.model_name = job_config.model.name

    def convert(self, model: nn.Module) -> nn.Module:
        self._validate_compatibility()

        try:
            count = self._patch_cls.apply(model, self.model_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to apply patch '{self._patch_name}' : {e}"
            ) from e
        if count > 0:
            logger.info(
                f"[NPU-CONVERTER] Applied '{self._patch_name}' : {count} replacements"
            )
        else:
            logger.warning(f"[NPU-CONVERTER] Applied no '{self._patch_name}' converter")
        return model

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]):
        pass

    def _validate_compatibility(self):
        if not self._patch_cls.is_compatible(self.job_config, self.model_name):
            raise ValueError(
                f"Patch '{self._patch_name}' is NOT compatible with model '{self.model_name}' \n"
                f"Supported models: {self._supported_models}"
            )
