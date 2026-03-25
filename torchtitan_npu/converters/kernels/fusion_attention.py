# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu

from ..base_converter import BaseConverter
from ..convert_utils import replace_modules
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)


class NPUFusionAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ) -> torch.Tensor:
        _, n_heads, _, dim = q.shape
        causal_mask = torch.triu(
            torch.ones((2048, 2048), dtype=torch.bool, device=q.device), diagonal=1
        )
        output = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            n_heads,
            "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=causal_mask,
            gen_mask_parallel=True,
            scale=scale if scale is not None else 1.0 / (dim**0.5),
            sparse_mode=2,
            # The debug parameter (sync) is a control switch for DSA to generate the dropout random number vector mask.
            # It defaults to False, which causes the dropout mask to be generated asynchronously on a separate stream,
            # leading to multi-stream failure in graph capture. Setting it to True enables synchronous (co-)generation,
            # resolving the multi-stream issue.
            sync=True,
        )[0]
        return output


def _create_npu_fusion_attention(model: nn.Module) -> nn.Module:
    """create npu_fusion_attention"""
    inner_attention = NPUFusionAttention()
    return inner_attention


@register_npu_converter("npu_fusion_attention")
class FusionAttentionKernel(BaseConverter):

    SUPPORTED_MODELS = {"llama3"}

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> nn.Module:

        count = 0
        count += replace_modules(
            model, r"ScaledDotProductAttentionWrapper", _create_npu_fusion_attention
        )

        # pyrefly: ignore [bad-return]
        return count
