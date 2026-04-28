# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torchtitan.distributed.context_parallel as titan_cp
import torchtitan.models.deepseek_v3 as titan_deepseekv3

from torchtitan_npu.models.deepseek_v32.infra.parallelize import apply_moe_ep_tp


# pyrefly: ignore [implicit-import]
_origin = titan_deepseekv3.infra.parallelize
_origin.parallelize_deepseekv3.__globals__["apply_moe_ep_tp"] = apply_moe_ep_tp
_original_parallelize = _origin.parallelize_deepseekv3


def _is_ulysses_config(job_config) -> bool:
    """
    Return True when DeepSeek V3 should route apply_cp_to_attention_module to Ulysses.
    """
    parallelism = getattr(job_config, "parallelism", None)
    if not getattr(parallelism, "enable_custom_context_parallel", False):
        return False
    cp_degree = getattr(parallelism, "context_parallel_degree", 1)
    return cp_degree > 1


@functools.wraps(_original_parallelize)
def _parallelize_deepseekv3_wrapper(model, parallel_dims, job_config):
    assert not (
        "npu_gmm" in job_config.model.converters
        and not parallel_dims.ep_enabled
        and parallel_dims.tp_enabled
    ), "npu_gmm is not supported when only tp is enabled. "

    if not _is_ulysses_config(job_config):
        return _original_parallelize(model, parallel_dims, job_config)

    npu_apply_cp = titan_cp.apply_cp_to_attention_module
    model_args = model.model_args

    def _apply_cp_ulysses(attention_modules, cp_mesh, attention_type):
        npu_apply_cp(
            attention_modules,
            cp_mesh,
            "ulysses",
            job_config=job_config,  # pyrefly: ignore [unexpected-keyword]
            model_args=model_args,  # pyrefly: ignore [unexpected-keyword]
        )

    _origin.parallelize_deepseekv3.__globals__[
        "apply_cp_to_attention_module"
    ] = _apply_cp_ulysses
    return _original_parallelize(model, parallel_dims, job_config)
