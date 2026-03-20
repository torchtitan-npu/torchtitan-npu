# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from tests.conftest import assert_tensor_finite, stable_randn
from torchtitan_npu.converters.kernels.fusion_attention import NPUFusionAttention


pytestmark = pytest.mark.smoke


def test_fusion_attention_wrapper_matches_expected_shape(npu_device):
    module = NPUFusionAttention().to(npu_device)
    q = stable_randn(1, 8, 128, 64, dtype=torch.bfloat16, device=npu_device)
    k = stable_randn(1, 8, 128, 64, dtype=torch.bfloat16, device=npu_device)
    v = stable_randn(1, 8, 128, 64, dtype=torch.bfloat16, device=npu_device)

    output = module(q, k, v)

    assert output.shape == q.shape
    assert_tensor_finite(output)


def test_fusion_attention_wrapper_supports_explicit_scale(npu_device):
    module = NPUFusionAttention().to(npu_device)
    q = stable_randn(1, 8, 128, 64, dtype=torch.bfloat16, device=npu_device)
    k = stable_randn(1, 8, 128, 64, dtype=torch.bfloat16, device=npu_device)
    v = stable_randn(1, 8, 128, 64, dtype=torch.bfloat16, device=npu_device)

    output = module(q, k, v, scale=1.0 / 8.0)

    assert output.shape == q.shape
    assert_tensor_finite(output)
