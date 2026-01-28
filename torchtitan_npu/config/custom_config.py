# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from torchtitan.config.job_config import (
    JobConfig as BaseJobConfig,
    Optimizer as BaseOptimizer,
    Parallelism as BaseParallelism
    )


@dataclass
class Optimizer(BaseOptimizer):
    swap_optimizer: bool = False
    """
    Whether to apply swap optimizer.
    This feature will offload the optimizer states to the host (CPU) during the forward and backward passes.
    During the optimizer.step(), it will load, update, and offload these states in slices.
    This pipelined approach significantly reduces GPU memory pressure during the optimizer step,
    making it highly beneficial for memory-intensive scenarios.
    More info (in Chinese): https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/swap-optimizer.md
    """

    swap_optimizer_times: int = 16
    """
    Specifies the number of slices for the pipelined swap_optimizer update.
    A higher value creates more, smaller slices, further reducing peak memory usage during the optimizer step.
    """


@dataclass
class Parallelism(BaseParallelism):
    enable_custom_context_parallel: bool = False
    """
    Whether to use a custom context manager for context parallel.
    If enable this, the 'custom_context_parallel_path' should be set correctly.
    """

    custom_context_parallel_path: str = ''
    """
    The path to custom context parallel context manager class.
    - The string must adhere to the format 'package.module.ClassName'.
    - The recommended custom class is a subclass of
    'torchtitan_npu.patches.distributed.custom_context_parallel.CustomContextParallelContext'

    Example string: 'torchtitan_npu.distributed.context_parallel.dsa_cp.AscendDSAContextParallelContext'
    """


@dataclass
class JobConfig(BaseJobConfig):
    optimizer: Optimizer = None
    parallelism: Parallelism = None