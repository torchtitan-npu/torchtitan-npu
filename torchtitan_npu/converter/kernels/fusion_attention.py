# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu

from ..registry import (
    BaseKernel,
    KernelType,
    replace_modules,
)

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
        causal_mask = torch.triu(torch.ones((2048, 2048), dtype=torch.bool, device=q.device), diagonal=1)
        output = torch_npu.npu_fusion_attention(
            q, k, v, n_heads, "BNSD",
            pse=None,
            padding_mask=None,
            atten_mask=causal_mask,
            gen_mask_parallel=True,
            scale=scale if scale is not None else 1.0 / (dim ** 0.5),
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


class FusionAttentionKernel(BaseKernel):
    kernel_type = KernelType.FUSIONATTEN

    @classmethod
    def apply(cls, model: nn.Module, **kwargs) -> nn.Module:

        count = 0
        count += replace_modules(model, r"ScaledDotProductAttentionWrapper", _create_npu_fusion_attention)

        logger.info(f" [FusionAttentionKernel] Applied {count} replacement(s).")
        return model