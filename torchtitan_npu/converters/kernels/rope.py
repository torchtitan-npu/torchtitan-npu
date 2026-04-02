# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu

from ..base_converter import BaseConverter
from ..convert_utils import find_functions, replace_functions
from ..registry import register_npu_converter

from .rope_broadcast import reshape_for_broadcast as _reshape_freqs_cis_for_broadcast

logger = logging.getLogger(__name__)


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


def npu_apply_rotary_emb_deepseek(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    interleaved: bool = True,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    if not interleaved:
        dtype = x.dtype
        shape = x.shape
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
        x_complex = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
        freqs_b = _reshape_freqs_cis_for_broadcast(freqs_cis, x_complex, positions)
        y = torch.view_as_real(x_complex * freqs_b).flatten(3)
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
        return y.to(dtype)

    dtype = x.dtype
    x_work = x.float()

    x_ref = torch.view_as_complex(
        x_work.reshape(*x_work.shape[:-1], x_work.shape[-1] // 2, 2)
    )
    freqs_b = _reshape_freqs_cis_for_broadcast(freqs_cis, x_ref, positions)
    cos, sin = _prepare_cos_sin_from_complex(
        freqs_b, x_work.dtype, already_broadcast=True
    )
    y = torch_npu.npu_rotary_mul(x_work, cos, sin, rotary_mode="interleave")
    return y.to(dtype)


def npu_apply_rotary_emb_deepseek_v3(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_b = _reshape_freqs_cis_for_broadcast(freqs_cis, x_complex, positions)
    y = torch.view_as_real(x_complex * freqs_b).flatten(3)
    return y.to(dtype)


def npu_apply_rotary_emb_llama(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    freqs_cis = _reshape_freqs_cis_for_broadcast(freqs_cis, xq_complex, positions)
    cos, sin = _prepare_cos_sin_from_complex(
        freqs_cis, xq.dtype, already_broadcast=True
    )
    return torch_npu.npu_rotary_mul(
        xq, cos, sin, rotary_mode="interleave"
    ), torch_npu.npu_rotary_mul(
        xk, cos.to(xk.dtype), sin.to(xk.dtype), rotary_mode="interleave"
    )


def npu_apply_rotary_emb_qwen(
    xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim, seqlen = xq.shape[-1], xq.shape[1]
    rope_cache = rope_cache[:seqlen].view(1, seqlen, 1, -1)
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype)
    return torch_npu.npu_rotary_mul(xq, cos, sin), torch_npu.npu_rotary_mul(
        xk, cos.to(xk.dtype), sin.to(xk.dtype)
    )


@register_npu_converter("npu_rope")
class RoPEKernel(BaseConverter):

    MODEL_IMPL = {
        "deepseek_v32": npu_apply_rotary_emb_deepseek,
        "deepseek_v3": npu_apply_rotary_emb_deepseek_v3,
        "qwen3": npu_apply_rotary_emb_qwen,
        "_default": npu_apply_rotary_emb_llama,
    }

    @classmethod
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:
        target = "apply_rotary_emb"
        matches = find_functions(target, model=model)
        if not matches:
            return 0

        impl = cls.get_impl_cls(model_name) or cls.MODEL_IMPL["_default"]

        count = replace_functions(target, impl, model=model)

        return count
