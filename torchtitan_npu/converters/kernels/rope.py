# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch_npu

from ..base_converter import BaseConverter
from ..convert_utils import find_functions, replace_functions
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)


def _prepare_cos_sin_from_complex(freqs_cis: torch.Tensor, dtype: torch.dtype):
    cos = freqs_cis.real.repeat_interleave(2, dim=-1)
    sin = freqs_cis.imag.repeat_interleave(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2).to(dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(dtype)
    return cos, sin


def _prepare_cos_sin_from_cache(rope_cache: torch.Tensor, x: torch.Tensor):
    head_dim = x.shape[-1]
    seqlen = x.shape[1]
    rope_cache = rope_cache[:seqlen].view(1, seqlen, 1, -1)
    cos = rope_cache[..., :head_dim].to(dtype=x.dtype)
    sin = rope_cache[..., head_dim:].to(dtype=x.dtype)
    return cos, sin


def npu_apply_rotary_emb_deepseek(
    x: torch.Tensor, freqs_cis: torch.Tensor, interleaved: bool = True
) -> torch.Tensor:
    dtype = x.dtype
    shape = x.shape

    x = x.float()
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous().view(*shape)

    cos, sin = _prepare_cos_sin_from_complex(freqs_cis, x.dtype)
    y = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")

    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)

    return y.to(dtype)


def npu_apply_rotary_emb_llama(
    xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _prepare_cos_sin_from_complex(freqs_cis, xq.dtype)
    return torch_npu.npu_rotary_mul(xq, cos, sin), torch_npu.npu_rotary_mul(
        xk, cos.to(xk.dtype), sin.to(xk.dtype)
    )


def npu_apply_rotary_emb_qwen(
    xq: torch.Tensor, xk: torch.Tensor, rope_cache: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos, sin = _prepare_cos_sin_from_cache(rope_cache, xq)
    return torch_npu.npu_rotary_mul(xq, cos, sin), torch_npu.npu_rotary_mul(
        xk, cos.to(xk.dtype), sin.to(xk.dtype)
    )


@register_npu_converter("npu_rope")
class RoPEKernel(BaseConverter):

    MODEL_IMPL = {
        "deepseek_v3": npu_apply_rotary_emb_deepseek,
        "deepseek_v32": npu_apply_rotary_emb_deepseek,
        "qwen3": npu_apply_rotary_emb_qwen,
        "default": npu_apply_rotary_emb_llama,
    }

    @classmethod
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> nn.Module:
        target = "apply_rotary_emb"
        matches = find_functions(target, model=model)
        if not matches:
            return model

        impl = cls.get_impl_cls(model_name)

        count = replace_functions(target, impl, model=model)

        return count
