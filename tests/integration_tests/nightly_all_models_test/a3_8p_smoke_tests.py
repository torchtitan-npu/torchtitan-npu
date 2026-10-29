# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the BSD-style license found in the LICENSE file.

"""Fast 8P distributed-launch smoke; intentionally does not load DSV4 weights."""
from tests.integration_tests import OverrideDefinitions


def build_test_list() -> list[OverrideDefinitions]:
    return [OverrideDefinitions(
        test_name="dispatch_smoke_8p",
        test_descr="Lite Actions dispatch smoke: single-node 8P HCCL/DP handshake",
        ngpu=8,
        nnodes=1,
        env_vars={
            "ASCEND_SET_ENV_PATH": "/mnt/share/Ascend/20260805101249091/ascend-toolkit/latest/set_env.sh",
            "HF_ASSETS_PATH": "/mnt/share/models/DeepSeek-V4-Flash-bf16",
            "CKPT_INIT_LOAD_PATH": "/mnt/share/dsv4_ckpt_8rank",
        },
        check_loss=False,
        timeout=75,
    )]
