# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torchtitan.models.deepseek_v3 as titan_deepseekv3

from torchtitan_npu.models.deepseek_v32.infra.parallelize import apply_moe_ep_tp


# pyrefly: ignore [implicit-import]
_origin = titan_deepseekv3.infra.parallelize
_origin.parallelize_deepseekv3.__globals__["apply_moe_ep_tp"] = apply_moe_ep_tp
_original_parallelize = _origin.parallelize_deepseekv3


@functools.wraps(_original_parallelize)
def _parallelize_deepseekv3_wrapper(model, parallel_dims, job_config):
    assert not (
        "npu_gmm" in job_config.model.converters
        and not parallel_dims.ep_enabled
        and parallel_dims.tp_enabled
    ), "npu_gmm is not supported when only tp is enabled. "
    return _original_parallelize(model, parallel_dims, job_config)
