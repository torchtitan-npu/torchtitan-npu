# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import time
from typing import Any

import torchtitan.components.metrics as metrics_module
import torchtitan.tools.utils as titan_tools_utils
from torchtitan.tools.logging import logger


def load_class_from_string(class_path: str):
    """Dynamically load class according to a string."""
    try:
        module_path, class_name = class_path.rsplit(".", 1)
    except ValueError as e:
        raise ValueError(
            f"Class string path error: '{class_path}', need to be 'module.path.ClassName'"
        ) from e

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}") from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(
            f"Module '{module_path}' does not have class '{class_name}'"
        ) from e

    return cls


_original_get_peak_flops = titan_tools_utils.get_peak_flops


def _patched_get_peak_flops(device_name: str) -> float:
    if "Ascend910_9392" in device_name:
        return 353.8944e12  # total: 376T Cube: 353T
    elif "Ascend910B1" in device_name:
        return 373.88e12  # total: 400T Cube: 373T
    elif "Ascend910B2" in device_name:
        return 353.8944e12  # total: 376T Cube: 353T
    elif "Ascend910B3" in device_name:
        return 294.912e12  # total: 313T Cube: 294T
    elif "Ascend910B4" in device_name:
        return 245.76e12  # total: 280T Cube: 245T
    return _original_get_peak_flops(device_name)


# Adapted from https://github.com/pytorch/torchtitan/blob/v0.2.1/torchtitan/components/metrics.py
def _patched_metrics_processor_log(
    self,
    step: int,
    global_avg_loss: float,
    global_max_loss: float,
    grad_norm: float,
    extra_metrics: dict[str, Any] | None = None,
):
    """Patched MetricsProcessor.log with elapsed_time_per_step in console output."""
    assert self.num_flops_per_token > 0, "num_flops_per_token must be set"

    time_delta = time.perf_counter() - self.time_last_log

    # tokens per second per device, abbreviated as tps
    tps = self.ntokens_since_last_log / (
        time_delta * self.parallel_dims.non_data_parallel_size
    )
    # model FLOPS utilization
    # For its definition and calculation, please refer to the PaLM paper:
    # https://arxiv.org/abs/2204.02311
    mfu = 100 * self.num_flops_per_token * tps / self.gpu_peak_flops
    tflops = self.num_flops_per_token * tps / 1e12

    time_end_to_end = time_delta / self.job_config.metrics.log_freq
    time_data_loading = sum(self.data_loading_times) / len(self.data_loading_times)
    time_data_loading_pct = 100 * sum(self.data_loading_times) / time_delta

    device_mem_stats = self.device_memory_monitor.get_peak_stats()

    metrics = {
        "loss_metrics/global_avg_loss": global_avg_loss,
        "loss_metrics/global_max_loss": global_max_loss,
        "grad_norm": grad_norm,
        "throughput(tps)": tps,
        "tflops": tflops,
        "mfu(%)": mfu,
        "time_metrics/end_to_end(s)": time_end_to_end,
        "time_metrics/data_loading(s)": time_data_loading,
        "time_metrics/data_loading(%)": time_data_loading_pct,
        "memory/max_active(GiB)": device_mem_stats.max_active_gib,
        "memory/max_active(%)": device_mem_stats.max_active_pct,
        "memory/max_reserved(GiB)": device_mem_stats.max_reserved_gib,
        "memory/max_reserved(%)": device_mem_stats.max_reserved_pct,
        "memory/num_alloc_retries": device_mem_stats.num_alloc_retries,
        "memory/num_ooms": device_mem_stats.num_ooms,
    }

    if extra_metrics:
        metrics.update(extra_metrics)

    self.logger.log(metrics, step)

    color = self.color
    logger.info(
        f"{color.red}step: {step:2}  "
        f"{color.green}loss: {global_avg_loss:8.5f}  "
        f"{color.orange}grad_norm: {grad_norm:7.4f}  "
        f"{color.turquoise}memory: {device_mem_stats.max_reserved_gib:5.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)  "
        f"{color.blue}tps: {round(tps):,}  "
        f"{color.cyan}tflops: {tflops:,.2f}  "
        f"{color.magenta}mfu: {mfu:.2f}%  "
        f"{color.yellow}elapsed_time_per_step: "
        f"{time_end_to_end:.3f}s{color.reset}"
    )

    self.ntokens_since_last_log = 0
    self.data_loading_times.clear()
    self.time_last_log = time.perf_counter()
    self.device_memory_monitor.reset_peak_stats()


# patch for step time print
metrics_module.MetricsProcessor.log = _patched_metrics_processor_log

# patch for Ascend peak flops
titan_tools_utils.get_peak_flops = _patched_get_peak_flops
