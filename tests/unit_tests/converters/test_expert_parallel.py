# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.distributed.expert_parallel import ExpertParallel

from torchtitan_npu.converters.kernels.expert_parallel import (
    _npu_moe_token_combine,
    _npu_moe_token_dispatch,
    ExpertParallelConverter,
)


def test_expert_parallel_converter_apply():

    original_token_dispatch = ExpertParallel._token_dispatch
    original_token_combine = ExpertParallel._token_combine

    counts = ExpertParallelConverter.apply(None, None)

    assert counts == 2
    assert ExpertParallel._token_dispatch == _npu_moe_token_dispatch
    assert ExpertParallel._token_combine == _npu_moe_token_combine
    assert ExpertParallel._token_dispatch != original_token_dispatch
    assert ExpertParallel._token_combine != original_token_combine
