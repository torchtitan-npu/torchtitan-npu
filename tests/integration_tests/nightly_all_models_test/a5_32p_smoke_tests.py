# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the BSD-style license found in the LICENSE file.

"""A5 smoke tests routed to the yellow Lite Actions worker."""
from tests.integration_tests import OverrideDefinitions


def build_test_list() -> list[OverrideDefinitions]:
    return [
        OverrideDefinitions(
            ci_kind="npu_smoke",
            test_name=name,
            test_descr=f"A5 {nodes * 8}P HCCL/DP dispatcher smoke",
            nnodes=nodes,
            ngpu=8,
            ci_overrides={
                "worker": "yellow",
                "target_pool": "a5_shared_yellow",
                "docker": "vllm-kimi-wangjinyin",
            },
            env_vars={"ASCEND_SET_ENV_PATH": "/usr/local/Ascend/ascend-toolkit/set_env.sh"},
            check_loss=False,
            timeout=timeout,
        )
        for name, nodes, timeout in (
            ("dispatch_smoke_a5_8p", 1, 120),
            ("dispatch_smoke_32p", 4, 150),
        )
    ]
