# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu
from torchtitan.models.qwen3.model.model import (
    reshape_for_broadcast as reshape_for_broadcast_qwen,
)

from ..base_converter import BaseConverter
from ..convert_utils import find_functions, replace_functions
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:

    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    if positions is None:
        freqs_cis = freqs_cis[0:seqlen]
        assert freqs_cis.shape == (seqlen, x.shape[-1] // 2)
        shape = [1, seqlen, 1, freqs_cis.shape[-1]]
        return freqs_cis.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        freqs_cis_real = torch.view_as_real(freqs_cis)
        freqs_cis_real = freqs_cis_real[positions.squeeze(0)]
        freqs_cis = torch.view_as_complex(freqs_cis_real)
        assert freqs_cis.shape == (seqlen, x.shape[-1] // 2)
        shape = [1, seqlen, 1, freqs_cis.shape[-1]]
        return freqs_cis.view(*shape)
    else:
        assert positions.shape == (x.shape[0], seqlen)
        freqs_cis_real = torch.view_as_real(freqs_cis)
        freqs_cis_real = freqs_cis_real[positions]
        freqs_cis = torch.view_as_complex(freqs_cis_real)
        shape = [x.shape[0], seqlen, 1, freqs_cis.shape[-1]]
        return freqs_cis.view(*shape)


def _prepare_cos_sin_from_complex(
    freqs_cis: torch.Tensor, dtype: torch.dtype, already_broadcast: bool = False
):
    cos = freqs_cis.real.repeat_interleave(2, dim=-1)
    sin = freqs_cis.imag.repeat_interleave(2, dim=-1)
    if not already_broadcast:
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)
    cos = cos.to(dtype)
    sin = sin.to(dtype)
    return cos, sin


def _prepare_cos_sin_from_cache(rope_cache: torch.Tensor, x: torch.Tensor):
    head_dim = x.shape[-1]
    seqlen = x.shape[1]
    rope_cache = rope_cache[:seqlen].view(1, seqlen, 1, -1)
    cos = rope_cache[..., :head_dim].to(dtype=x.dtype)
    sin = rope_cache[..., head_dim:].to(dtype=x.dtype)
    return cos, sin


def npu_apply_rotary_emb_deepseek(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
    interleaved: bool = True,
) -> torch.Tensor:

    dtype = x.dtype
    shape = x.shape

    x = x.float()
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous().view(*shape)

    freqs_cis = reshape_for_broadcast(freqs_cis, x, positions)[0].unsqueeze(0)

    cos, sin = _prepare_cos_sin_from_complex(freqs_cis, x.dtype, already_broadcast=True)
    y = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")

    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)

    return y.to(dtype)


def npu_apply_rotary_emb_deepseek_v4(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()

    freqs_real = torch.view_as_real(freqs_cis)
    cos = freqs_real[..., 0]  # (seq_len, head_dim/2)
    sin = freqs_real[..., 1]  # (seq_len, head_dim/2)

    if inverse:
        sin = -sin

    r1 = torch.stack([cos, cos], dim=-1).flatten(-2)  # (seq_len, head_dim)
    r2 = torch.stack([sin, sin], dim=-1).flatten(-2)  # (seq_len, head_dim)

    input_3d = x.ndim == 3
    if input_3d:
        # (batch, seq_len, head_dim) -> (batch, seq_len, 1, head_dim)
        x_4d = x.unsqueeze(2)
    elif x.ndim == 4:
        x_4d = x
    else:
        raise ValueError(f"Input tensor must be 3D or 4D, got {x.ndim}D")

    # (seq_len, head_dim) -> (1, seq_len, 1, head_dim)
    r1 = r1.view(1, r1.size(0), 1, r1.size(1))
    r2 = r2.view(1, r2.size(0), 1, r2.size(1))

    y = torch_npu.npu_rotary_mul(x_4d, r1, r2, rotary_mode="interleave")

    if input_3d:
        y = y.squeeze(2)

    return y.to(dtype)


def npu_apply_rotary_emb_llama(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:

    xq_ = xq.float()
    xk_ = xk.float()

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_, positions)[0].unsqueeze(0)

    cos, sin = _prepare_cos_sin_from_complex(
        freqs_cis, xq_.dtype, already_broadcast=True
    )
    return torch_npu.npu_rotary_mul(xq_, cos, sin, rotary_mode="interleave").type_as(
        xq
    ), torch_npu.npu_rotary_mul(
        xk_, cos.to(xk_.dtype), sin.to(xk_.dtype), rotary_mode="interleave"
    ).type_as(
        xk
    )


def npu_apply_rotary_emb_qwen(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope_cache = reshape_for_broadcast_qwen(rope_cache, xq, positions)

    cos, sin = _prepare_cos_sin_from_cache(rope_cache, xq)

    return torch_npu.npu_rotary_mul(xq, cos, sin), torch_npu.npu_rotary_mul(
        xk, cos, sin
    )


@register_npu_converter("npu_rope")
class RoPEKernel(BaseConverter):

    MODEL_IMPL = {
        "deepseek_v3": npu_apply_rotary_emb_deepseek,
        "deepseek_v32": npu_apply_rotary_emb_deepseek,
        "deepseek_v4": npu_apply_rotary_emb_deepseek_v4,
        "qwen3": npu_apply_rotary_emb_qwen,
        "_default": npu_apply_rotary_emb_llama,
    }

    @classmethod
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:
        target = "apply_rotary_emb"
        matches = find_functions(target, model=model)
        if not matches:
            return 0

        impl = cls.get_impl_cls(model_name)

        # pyrefly: ignore [bad-argument-type]
        count = replace_functions(target, impl, model=model)

        return count
