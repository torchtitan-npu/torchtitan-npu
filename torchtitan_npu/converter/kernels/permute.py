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
    replace_methods,
)

logger = logging.getLogger(__name__)


def _npu_moe_forward(self, x):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)
    num_tokens = x.shape[0]
    top_scores, selected_experts_indices, num_tokens_per_expert = self.router(x, self.expert_bias)

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    if selected_experts_indices.dim() == 1:
        expert_indices_2d = selected_experts_indices.view(num_tokens, -1)
        top_scores_2d = top_scores.view(num_tokens, -1)
    else:
        expert_indices_2d = selected_experts_indices
        top_scores_2d = top_scores

    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(
        x, expert_indices_2d.to(torch.int32)
    )
    top_scores_sorted = top_scores_2d.flatten()[sorted_indices]

    if self.score_before_experts:
        routed_input = (routed_input.float() * top_scores_sorted.reshape(-1, 1)).to(x.dtype)

    routed_output = self.experts(routed_input, num_tokens_per_expert)
    out = self.shared_experts(x) if self.shared_experts is not None else torch.zeros_like(x)

    if not self.score_before_experts:
        routed_output = (routed_output.float() * top_scores_sorted.reshape(-1, 1)).to(x.dtype)

    probs = torch.ones(num_tokens, top_scores_2d.shape[1], device=x.device, dtype=routed_output.dtype)
    unpermuted = torch_npu.npu_moe_token_unpermute(
        routed_output, sorted_indices.to(torch.int32), probs
    )
    return (out + unpermuted).reshape(bs, slen, dim)


class PermuteKernel(BaseKernel):

    kernel_type = KernelType.PERMUTE
    MOE_PACKAGE = "torchtitan.models.moe"

    @classmethod
    def apply(cls, model: nn.Module, **kwargs) -> nn.Module:
        pkg = cls.MOE_PACKAGE

        count = replace_methods("MoE", "forward", _npu_moe_forward, package=pkg)

        logger.info(f"  [Permute] Applied {count} replacement(s)")
        return model