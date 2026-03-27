# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os

import torch
import torchtitan.train as train_module

from torchtitan.config import Profiling as ProfilingConfig
from torchtitan.tools.logging import logger


def is_profile_enabled(profiling_config: ProfilingConfig) -> bool:
    """Check if profiling is enabled for the current rank."""
    if not profiling_config.enable_profiling:
        return False
    # pyrefly: ignore [missing-attribute]
    profile_ranks = profiling_config.profile_ranks
    if profile_ranks == [-1]:
        return True
    return torch.distributed.get_rank() in profile_ranks


def _log_profiling_config(rank: int, profiling_config: ProfilingConfig) -> None:
    """Log profiling configuration for the current rank."""
    logger.info(
        f"Profiling enabled for rank {rank} with config: "
        # pyrefly: ignore [missing-attribute]
        f"profile_step_start={profiling_config.profile_step_start}, "
        # pyrefly: ignore [missing-attribute]
        f"profile_step_end={profiling_config.profile_step_end}, "
        # pyrefly: ignore [missing-attribute]
        f"profile_ranks={profiling_config.profile_ranks}, "
        # pyrefly: ignore [missing-attribute]
        f"profile_record_shapes={profiling_config.profile_record_shapes}, "
        # pyrefly: ignore [missing-attribute]
        f"profile_with_memory={profiling_config.profile_with_memory}, "
        # pyrefly: ignore [missing-attribute]
        f"profile_with_stack={profiling_config.profile_with_stack}, "
        f"profile_freq={profiling_config.profile_freq}, "
        f"profiler_active={profiling_config.profiler_active}, "
        f"profiler_warmup={profiling_config.profiler_warmup}"
    )


@contextlib.contextmanager
def maybe_enable_profiling(
    profiling_config: ProfilingConfig,
    *,
    global_step: int = 0,
    base_folder: str = "",
    leaf_folder: str = "",
):
    if not is_profile_enabled(profiling_config):
        yield None
        return

    # pyrefly: ignore [missing-attribute]
    if not torch.npu.is_available():
        raise RuntimeError("Only NPU is supported currently")
    import torch_npu

    trace_dir = os.path.join(base_folder, profiling_config.save_traces_folder)
    rank = torch.distributed.get_rank()
    _log_profiling_config(rank, profiling_config)

    # pyrefly: ignore [missing-attribute]
    profile_step_start = profiling_config.profile_step_start
    # pyrefly: ignore [missing-attribute]
    profile_step_end = profiling_config.profile_step_end
    if profile_step_start > 0 and profile_step_end > 0:
        active = profile_step_end - profile_step_start
        wait = profile_step_start - global_step - 2
        warmup = 1 if profile_step_start > global_step + 1 else 0
        repeat = 1
    else:
        profile_freq = profiling_config.profile_freq
        warmup = profiling_config.profiler_warmup
        active = profiling_config.profiler_active
        wait = profile_freq - (active + warmup)
        repeat = 0

    if wait < 0:
        logger.warning(
            f"profile_step_start must be greater than {global_step} + 2, "
            f"or profile_freq must be greater or equal to warmup + active, but "
            f"got wait={wait}, profiling will be skipped."
        )
        yield None
        return

    logger.info(f"Profiling active. Traces will be saved at {trace_dir}")
    os.makedirs(trace_dir, exist_ok=True)
    # pyrefly: ignore [missing-attribute]
    enable_online_parse = profiling_config.enable_online_parse
    if enable_online_parse:
        on_trace_ready_handler = torch_npu.profiler.tensorboard_trace_handler(trace_dir)
    else:
        os.environ["ASCEND_WORK_PATH"] = trace_dir
        on_trace_ready_handler = None
        logger.info(f"Online parsing disabled. ASCEND_WORK_PATH set to {trace_dir}")
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        aic_metrics=torch_npu.profiler.AiCMetrics.ArithmeticUtilization,
    )

    with torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=repeat
        ),
        on_trace_ready=on_trace_ready_handler,
        # pyrefly: ignore [missing-attribute]
        record_shapes=profiling_config.profile_record_shapes,
        # pyrefly: ignore [missing-attribute]
        profile_memory=profiling_config.profile_with_memory,
        # pyrefly: ignore [missing-attribute]
        with_stack=profiling_config.profile_with_stack,
        experimental_config=experimental_config,
    ) as torch_profiler:
        torch_profiler.step_num = global_step
        yield torch_profiler


train_module.maybe_enable_profiling = maybe_enable_profiling
