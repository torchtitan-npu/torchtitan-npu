# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest

try:
    from torch.testing._internal.distributed._tensor.common_dtensor import (
        DTensorTestBase,
        with_comms,
    )

    MULTI_RANK_AVAILABLE = True

    class FourRankMultiRankTestBase(DTensorTestBase):
        @property
        def world_size(self):
            return 4

except ImportError:
    MULTI_RANK_AVAILABLE = False
    FourRankMultiRankTestBase = object
    with_comms = None


def mark_multi_rank_nightly(test_obj):
    return pytest.mark.nightly(
        pytest.mark.skipif(
            os.getenv("RUN_MODEL_PARALLEL_MULTI_RANK", "false").lower() != "true",
            reason="multi-rank model-parallel smoke is reserved for explicit nightly/torchrun runs",
        )(test_obj)
    )
