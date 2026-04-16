# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import shutil
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from tests.conftest import assert_tensor_finite, stable_randn

from torchtitan_npu.converters.kernels.gmm import (
    _run_experts_grouped_mm,
    npu_grouped_experts_forward,
)


pytestmark = pytest.mark.smoke


def _expert_inputs(device, *, total_tokens=32, dim=128, hidden_dim=256, num_experts=4):
    x = stable_randn(total_tokens, dim, dtype=torch.bfloat16, device=device)
    w13 = stable_randn(
        num_experts, hidden_dim * 2, dim, dtype=torch.bfloat16, device=device
    )
    w2 = stable_randn(num_experts, dim, hidden_dim, dtype=torch.bfloat16, device=device)
    num_tokens_per_expert = torch.tensor(
        [total_tokens // num_experts] * num_experts,
        dtype=torch.int64,
        device=device,
    )
    return x, w13, w2, num_tokens_per_expert


def test_gmm_grouped_expert_kernel_forward(npu_device):
    x, w13, w2, num_tokens_per_expert = _expert_inputs(npu_device)

    output = _run_experts_grouped_mm(w13, w2, None, x, num_tokens_per_expert)

    assert output.shape == x.shape
    assert_tensor_finite(output)


def test_gmm_grouped_expert_kernel_backward(npu_device):
    x, w13, w2, num_tokens_per_expert = _expert_inputs(npu_device)
    x.requires_grad_()
    w13.requires_grad_()
    w2.requires_grad_()

    loss = _run_experts_grouped_mm(w13, w2, None, x, num_tokens_per_expert).sum()
    loss.backward()

    assert x.grad is not None
    assert w13.grad is not None
    assert w2.grad is not None


def test_gmm_grouped_experts_forward_uses_converter_layout(npu_device):
    if shutil.which("npuc") is None:
        pytest.skip(
            "GroupedExperts padding path requires npuc/triton compiler in the test environment"
        )

    x, w13, w2, num_tokens_per_expert = _expert_inputs(npu_device)
    module = SimpleNamespace(
        w2=nn.Parameter(w2),
        w13=nn.Parameter(w13),
        num_experts=w13.shape[0],
    )

    output = npu_grouped_experts_forward(module, x, num_tokens_per_expert)

    assert output.shape == x.shape
    assert_tensor_finite(output)
