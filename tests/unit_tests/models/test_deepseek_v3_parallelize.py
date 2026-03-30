# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_npu.models.deepseek_v3.infra.parallelize import apply_moe_ep_tp


def test_patch_applied_to_original_module():
    try:
        import torchtitan.models.deepseek_v3 as titan_deepseekv3

        assert hasattr(titan_deepseekv3.infra.parallelize, "apply_moe_ep_tp")

        patched_func = (
            titan_deepseekv3.infra.parallelize.parallelize_deepseekv3.__globals__.get(
                "apply_moe_ep_tp"
            )
        )
        assert patched_func is not None
        assert patched_func is apply_moe_ep_tp
    except ImportError:
        pass
