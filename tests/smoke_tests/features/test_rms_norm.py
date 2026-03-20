# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke tests for RMS Norm operator.

Requires NPU environment.
"""

import pytest
import torch

from tests.conftest import assert_tensor_finite, stable_randn

pytestmark = pytest.mark.smoke


def test_rms_norm_forward(npu_device):
    import torch_npu

    batch_size = 2
    seq_len = 128
    hidden_dim = 64

    x = stable_randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=npu_device)
    weight = torch.ones(hidden_dim, dtype=torch.float32, device=npu_device)
    eps = 1e-6

    try:
        output = torch_npu.npu_rms_norm(x, weight, eps)[0]

        assert output.shape == x.shape, f"Output shape {output.shape} should match input {x.shape}"
        assert_tensor_finite(output, "Output should be finite")

    except AttributeError:
        pytest.skip("npu_rms_norm not available in current torch_npu version")


def test_rms_norm_eps(npu_device):
    import torch_npu

    batch_size = 2
    seq_len = 64
    hidden_dim = 32

    x = stable_randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=npu_device)
    weight = torch.ones(hidden_dim, dtype=torch.float32, device=npu_device)

    eps_values = [1e-5, 1e-6, 1e-8]

    try:
        for eps in eps_values:
            output = torch_npu.npu_rms_norm(x, weight, eps)[0]
            assert_tensor_finite(output, f"Output should be finite for eps={eps}")

    except AttributeError:
        pytest.skip("npu_rms_norm not available")


def test_rms_norm_gradient(npu_device):
    import torch_npu

    batch_size = 2
    seq_len = 64
    hidden_dim = 32

    x = stable_randn(batch_size, seq_len, hidden_dim, dtype=torch.float32, device=npu_device, requires_grad=True)
    weight = torch.ones(hidden_dim, dtype=torch.float32, device=npu_device, requires_grad=True)
    eps = 1e-6

    try:
        output = torch_npu.npu_rms_norm(x, weight, eps)[0]
        loss = output.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient should exist"
        assert weight.grad is not None, "Weight gradient should exist"

    except AttributeError:
        pytest.skip("npu_rms_norm not available")


