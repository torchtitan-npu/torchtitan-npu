# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fast 16P distributed-launch smoke; intentionally does not load DSV4 weights."""
from tests.integration_tests import OverrideDefinitions


def build_test_list() -> list[OverrideDefinitions]:
    return [OverrideDefinitions(
        ci_kind="npu_smoke",
        test_name="dispatch_smoke_16p",
        test_descr="Lite Actions dispatch smoke: two-node 16P HCCL/DP handshake",
        ngpu=8,
        nnodes=2,
        env_vars={
            "ASCEND_SET_ENV_PATH": "/usr/local/Ascend/ascend-toolkit/set_env.sh",
        },
        check_loss=False,
        timeout=75,
    )]
