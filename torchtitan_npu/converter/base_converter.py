# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Union, List, Optional, Type

import torch.nn as nn

from torchtitan.config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.protocols.model_converter import ModelConverter

from .registry import KernelType, KERNEL_REGISTRY, BaseKernel

logger = logging.getLogger(__name__)


class BaseNPUConverter(ModelConverter):
    """
    NPU Base Converter
    """

    kernel_type: Optional[KernelType] = None
    kernel_name: str = "NPU"

    def __init__(self, job_config: JobConfig, parallel_dims: ParallelDims):
        self.job_config = job_config
        self.parallel_dims = parallel_dims
        from .kernel_converter import _ensure_kernels_loaded
        _ensure_kernels_loaded()

        # filtering config
        self._skip_types: set[KernelType] = set()
        self._parse_filter_config()

    def convert(self, model: nn.Module) -> None:
        """Implement ModelConverter Protocol"""
        kernels = self._get_kernels()

        if not kernels:
            logger.warning(f"[NPU] No kernels to apply for {self.kernel_name}")
            return

        logger.info(f"[NPU] {self.kernel_name}: applying {len(kernels)} kernel(s)")

        applied = 0
        failed = 0

        for kernel in kernels:
            name = kernel.kernel_type.name if kernel.kernel_type else kernel.__name__
            try:
                logger.info(f"[NPU][{name}] Applying...")
                kernel.apply(model)
                applied += 1
            except Exception as e:
                logger.error(f"[NPU][{name}] Failed: {e}")
                failed += 1

        logger.info(f"[NPU] Complete: {applied} applied, {failed} failed")

    def post_optimizer_hook(self, model: Union[nn.Module, List[nn.Module]]) -> None:
        pass
    
    def _parse_filter_config(self) -> None:
        if not hasattr(self.job_config, 'npu'):
            return

        npu_config = self.job_config.npu
        if hasattr(npu_config, 'skip_kernels'):
            self._skip_types = {KernelType[name] for name in npu_config.skip_kernels}

    def _get_kernels(self) -> list[Type[BaseKernel]]:
        """Acquire kernel list"""
        if self.kernel_type is not None:
            # Single kernel Mode
            kernel = KERNEL_REGISTRY.get(self.kernel_type)
            return [kernel] if kernel else []

        # Multi kernel Mode
        kernels = KERNEL_REGISTRY.get_all()

        if self._skip_types:
            kernels = [k for k in kernels if k.kernel_type not in self._skip_types]

        return kernels