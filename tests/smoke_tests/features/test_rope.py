# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
from torchtitan.models.deepseek_v3.model.model import (
    apply_rotary_emb as deepseek_apply_rotary_emb,
    precompute_freqs_cis as deepseek_precompute_freqs_cis,
)
from torchtitan.models.llama3.model.model import (
    apply_rotary_emb as llama_apply_rotary_emb,
    precompute_freqs_cis as llama_precompute_freqs_cis,
)
from torchtitan.models.qwen3.model.model import (
    apply_rotary_emb as qwen_apply_rotary_emb,
    precompute_rope_cache as qwen_precompute_rope_cache,
)

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


def _assert_tensors_close(
    expected: torch.Tensor,
    actual: torch.Tensor,
    message_prefix: str,
    *,
    rtol: float,
    atol: float,
):
    assert torch.allclose(
        expected.float(), actual.float(), rtol=rtol, atol=atol
    ), f"{message_prefix}: max_diff={torch.max(torch.abs(expected.float() - actual.float())).item()}"


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


def test_npu_apply_rotary_emb_llama_precision(npu_device):
    xq = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    xk = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    freqs_cis = llama_precompute_freqs_cis(64, 128).to(npu_device)

    expected_q, expected_k = llama_apply_rotary_emb(xq, xk, freqs_cis)
    actual_q, actual_k = npu_apply_rotary_emb_llama(xq, xk, freqs_cis)

    assert expected_q.shape == actual_q.shape
    assert expected_k.shape == actual_k.shape
    _assert_tensors_close(
        expected_q, actual_q, "Query output mismatch", rtol=1e-5, atol=1e-5
    )
    _assert_tensors_close(
        expected_k, actual_k, "Key output mismatch", rtol=1e-5, atol=1e-5
    )


def test_npu_apply_rotary_emb_qwen_precision(npu_device):
    xq = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    xk = stable_randn(2, 128, 8, 64, dtype=torch.float32, device=npu_device)
    rope_cache = qwen_precompute_rope_cache(64, 128).to(npu_device)

    expected_q, expected_k = qwen_apply_rotary_emb(xq, xk, rope_cache)
    actual_q, actual_k = npu_apply_rotary_emb_qwen(xq, xk, rope_cache)

    assert expected_q.shape == actual_q.shape
    assert expected_k.shape == actual_k.shape
    _assert_tensors_close(
        expected_q, actual_q, "Query output mismatch", rtol=1e-5, atol=1e-5
    )
    _assert_tensors_close(
        expected_k, actual_k, "Key output mismatch", rtol=1e-5, atol=1e-5
    )


def test_npu_apply_rotary_emb_deepseek_precision(npu_device):
    x = stable_randn(
        2,
        128,
        8,
        64,
        dtype=torch.float32,
        device=npu_device,
    )
    model_args = DeepSeekV3ModelArgs(max_seq_len=128, qk_rope_head_dim=64)
    freqs_cis = deepseek_precompute_freqs_cis(model_args).to(npu_device)

    expected = deepseek_apply_rotary_emb(x, freqs_cis)
    actual = npu_apply_rotary_emb_deepseek(x, freqs_cis)

    assert expected.shape == actual.shape

    _assert_tensors_close(expected, actual, "Output mismatch", rtol=1e-5, atol=1e-5)
