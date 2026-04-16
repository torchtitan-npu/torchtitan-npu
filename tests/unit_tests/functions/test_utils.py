# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

from torchtitan_npu.converters.kernels import rope as rope_kernel
from torchtitan_npu.tools.weight_utils import detect_input_format_by_path


def test_detect_input_format_by_path_defaults_to_hf(tmp_path):
    assert detect_input_format_by_path(str(tmp_path)) == "hf"


def test_detect_input_format_by_path_recognizes_dcp_markers(tmp_path):
    (tmp_path / ".metadata").write_text("metadata")

    assert detect_input_format_by_path(str(tmp_path)) == "dcp"


def test_npu_apply_rotary_emb_deepseek_prepares_interleaved_cos_sin(monkeypatch):
    captured = {}
    x = torch.randn(2, 8, 4, 32, dtype=torch.bfloat16)
    freqs_cis = torch.randn(8, 16, dtype=torch.complex64)

    def fake_rotary_mul(tensor, cos, sin, rotary_mode="half"):
        captured["cos"] = cos
        captured["sin"] = sin
        captured["rotary_mode"] = rotary_mode
        return tensor

    monkeypatch.setattr(rope_kernel.torch_npu, "npu_rotary_mul", fake_rotary_mul)

    result = rope_kernel.npu_apply_rotary_emb_deepseek(x, freqs_cis)

    assert result.shape == x.shape
    assert captured["cos"].shape == (1, 8, 1, 32)
    assert captured["sin"].shape == (1, 8, 1, 32)
    assert captured["cos"].dtype == torch.float32
    assert captured["sin"].dtype == torch.float32
    assert captured["rotary_mode"] == "interleave"


def test_npu_apply_rotary_emb_qwen_uses_input_sequence_length(monkeypatch):
    captured = []
    xq = torch.randn(2, 4, 8, 16, dtype=torch.float16)
    xk = torch.randn(2, 4, 8, 16, dtype=torch.float16)
    rope_cache = torch.randn(16, 32, dtype=torch.float32)

    def fake_rotary_mul(tensor, cos, sin):
        captured.append((cos, sin))
        return tensor

    monkeypatch.setattr(rope_kernel.torch_npu, "npu_rotary_mul", fake_rotary_mul)

    out_q, out_k = rope_kernel.npu_apply_rotary_emb_qwen(xq, xk, rope_cache)

    assert out_q.shape == xq.shape
    assert out_k.shape == xk.shape
    assert len(captured) == 2
    assert captured[0][0].shape == (1, 4, 1, 16)
    assert captured[0][1].shape == (1, 4, 1, 16)
    assert captured[0][0].dtype == xq.dtype
    assert captured[0][1].dtype == xq.dtype
