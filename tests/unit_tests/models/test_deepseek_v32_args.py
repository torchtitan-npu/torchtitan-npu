# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy
from types import SimpleNamespace

from torchtitan_npu.models.deepseek_v32 import deepseekv32_args


def test_update_from_config_keeps_grouped_mm_enabled():
    model_args = deepcopy(deepseekv32_args["tinymodel"])
    model_args.moe_args.use_grouped_mm = True
    assert model_args.moe_args.use_grouped_mm is True

    job_config = SimpleNamespace(
        training=SimpleNamespace(seq_len=4096),
        parallelism=SimpleNamespace(
            context_parallel_degree=1,
            expert_parallel_comm_backend="deepep",
        ),
        debug=SimpleNamespace(moe_force_load_balance=False),
    )

    model_args.update_from_config(job_config)

    assert model_args.moe_args.use_grouped_mm is True
