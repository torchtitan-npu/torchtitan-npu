# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from tests.conftest import assert_tensor_finite, stable_randn

from torchtitan_npu.converters.kernels.rope import (
    npu_apply_rotary_emb_deepseek,
    npu_apply_rotary_emb_llama,
    npu_apply_rotary_emb_qwen,
)

pytestmark = pytest.mark.smoke


def _complex_freqs(shape, device):
    real = stable_randn(*shape, dtype=torch.float32, device=device)
    imag = stable_randn(*shape, dtype=torch.float32, device=device)
    return torch.complex(real, imag)


def test_rope_deepseek(npu_device):
    x = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    freqs_cis = _complex_freqs((128, 32), npu_device)

    output = npu_apply_rotary_emb_deepseek(x, freqs_cis)

    assert output.shape == x.shape
    assert_tensor_finite(output)


def test_rope_llama(npu_device):
    xq = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    xk = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    freqs_cis = _complex_freqs((128, 32), npu_device)

    q_out, k_out = npu_apply_rotary_emb_llama(xq, xk, freqs_cis)

    assert q_out.shape == xq.shape
    assert k_out.shape == xk.shape


def test_rope_qwen(npu_device):
    xq = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    xk = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    rope_cache = stable_randn(128, 128, dtype=torch.float32, device=npu_device)

    q_out, k_out = npu_apply_rotary_emb_qwen(xq, xk, rope_cache)

    assert q_out.shape == xq.shape
    assert k_out.shape == xk.shape
