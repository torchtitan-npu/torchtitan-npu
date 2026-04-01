# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Literal

from torchtitan.config.job_config import (
    JobConfig as BaseJobConfig,
    Optimizer as BaseOptimizer,
    Parallelism as BaseParallelism,
    Profiling as BaseProfiling,
    Training as BaseTraining,
)


@dataclass
class Optimizer(BaseOptimizer):
    """
    Whether to apply swap optimizer.
    This feature will offload the optimizer states to the host (CPU) during the forward and backward passes.
    During the optimizer.step(), it will load, update, and offload these states in slices.
    This pipelined approach significantly reduces GPU memory pressure during the optimizer step,
    making it highly beneficial for memory-intensive scenarios.
    More info (in Chinese): https://gitcode.com/Ascend/MindSpeed/blob/master/docs/features/swap-optimizer.md
    """

    swap_optimizer: bool = False

    """
    Specifies the number of slices for the pipelined swap_optimizer update.
    A higher value creates more, smaller slices, further reducing peak memory usage during the optimizer step.
    """
    swap_optimizer_times: int = 16

    # Muon-specific parameters (used when name is "Muon")
    """
    Learning rate for Muon optimizer. If None, falls back to lr.
    """
    muon_lr: float | None = None

    """Momentum factor for Muon optimizer"""
    muon_momentum: float = 0.95

    """Whether to use Nesterov momentum for Muon"""
    muon_enable_nesterov: bool = True

    """Number of Newton-Schulz iteration steps for Muon"""
    muon_ns_steps: int = 5

    """
    Learning rate adjustment function for Muon. Options:
    - None or "original": Use sqrt(max(1, A/B)) ratio (muon_lr is used if specified)
    - "match_rms_adamw": Use 0.2 * sqrt(max(A, B)) ratio (muon_lr is ignored, uses base lr)
    """
    muon_adjust_lr_fn: Literal["original", "match_rms_adamw"] | None = "match_rms_adamw"


@dataclass
class Parallelism(BaseParallelism):
    """
    Whether to use a custom context manager for context parallel.
    If enable this, the 'custom_context_parallel_path' should be set correctly.
    """

    enable_custom_context_parallel: bool = False

    """
    The path to custom context parallel context manager class.
    - The string must adhere to the format 'package.module.ClassName'.
    - The recommended custom class is a subclass of
    'torchtitan_npu.patches.distributed.custom_context_parallel.CustomContextParallelContext'

    Example string: 'torchtitan_npu.distributed.context_parallel.dsa_cp.AscendDSAContextParallelContext'
    """
    custom_context_parallel_path: str = ""


@dataclass
class Training(BaseTraining):
    """
    Specifies the maximum proportion of NPU memory that PyTorch is allowed to occupy.
    The value ranges from 0.0 to 1.0, where 0.9 means PyTorch can use up to 90% of the total NPU memory.
    Adjusting this value helps control memory usage and avoid out-of-memory (OOM) errors on NPU devices.
    """

    torch_npu_memory_ratio: float = 1.0


@dataclass
class Profiling(BaseProfiling):
    """
    The step at which to start profiling.
    Profiling will begin at this step and continue for `profiler_active` steps.
    """

    profile_step_start: int = 0

    """
    The step at which to end profiling.
    If set to 0, will use profile_step_start + profiler_active.
    """
    profile_step_end: int = 0

    """
    List of ranks to profile, e.g., [0, 1, 2].
    Use [-1] to profile all ranks.
    Default is [-1] (all ranks).
    """
    profile_ranks: list[int] = field(default_factory=lambda: [-1])

    """
    Whether to record tensor shapes during profiling.
    """
    profile_record_shapes: bool = True

    """
    Whether to profile memory usage.
    """
    profile_with_memory: bool = False

    """
    Whether to record stack traces during profiling.
    """
    profile_with_stack: bool = False

    """
    Whether to enable online parsing of profiling data.
    If disabled, on_trace_ready will be set to None and ASCEND_WORK_PATH environment
    variable will be set to trace_dir for offline parsing.
    """
    enable_online_parse: bool = True


@dataclass
class JobConfig(BaseJobConfig):
    # pyrefly: ignore [bad-override]
    optimizer: Optimizer = field(default_factory=Optimizer)
    # pyrefly: ignore [bad-override]
    parallelism: Parallelism = field(default_factory=Parallelism)
    # pyrefly: ignore [bad-override]
    training: Training = field(default_factory=Training)
    # pyrefly: ignore [bad-override]
    profiling: Profiling = field(default_factory=Profiling)
