# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu

from ..base_converter import BaseConverter
from ..convert_utils import replace_methods
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)


def _npu_moe_forward_for_dsv32(self, x):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)

    (top_scores, selected_experts_indices, num_tokens_per_expert) = self.router(
        x, self.expert_bias
    )

    if self.shared_experts is not None:
        out = self.shared_experts(x)
    else:
        out = torch.zeros_like(x)

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    indices = selected_experts_indices.view(-1, self.reorderer.top_k)
    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x, indices)

    routed_output = self.experts(routed_input, num_tokens_per_expert)

    unpermuted = torch_npu.npu_moe_token_unpermute(
        routed_output,
        sorted_indices,
        # Mixing FP32 `topk_score` and BF16 `routed_output` causes
        # MoeTokenUnpermuteGrad to return NaN values. Cast the FP32
        # part to BF16 as a temporary workaround.
        top_scores.to(x.dtype),
    )
    return (out + unpermuted).reshape(bs, slen, dim)


def _npu_moe_forward_for_dsv4(self, x, input_ids):
    bs, slen, dim = x.shape
    x = x.view(-1, dim)
    input_ids_flat = input_ids.flatten() if input_ids is not None else None

    (top_scores, selected_experts_indices, num_tokens_per_expert) = self.router(
        x, input_ids_flat, self.expert_bias
    )

    with torch.no_grad():
        self.tokens_per_expert.add_(num_tokens_per_expert)

    indices = selected_experts_indices.view(-1, self.reorderer.top_k)
    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x, indices)

    routed_output = self.experts(routed_input, num_tokens_per_expert)

    if self.shared_experts is not None:
        out = self.shared_experts(x)
    else:
        out = torch.zeros_like(x)

    unpermuted = torch_npu.npu_moe_token_unpermute(
        routed_output,
        sorted_indices,
        # Mixing FP32 `topk_score` and BF16 `routed_output` causes
        # MoeTokenUnpermuteGrad to return NaN values. Cast the FP32
        # part to BF16 as a temporary workaround.
        top_scores.to(x.dtype),
    )
    return (out + unpermuted).reshape(bs, slen, dim)


@register_npu_converter("npu_permute")
class PermuteKernel(BaseConverter):

    # pyrefly: ignore [bad-assignment]
    MODEL_IMPL = {
        "deepseek_v3": ("torchtitan.models.moe", _npu_moe_forward_for_dsv32),
        "deepseek_v4": (
            "torchtitan_npu.models.deepseek_v4.model.moe",
            _npu_moe_forward_for_dsv4,
        ),
    }

    @classmethod
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:
        impl = cls.get_impl_cls(model_name)
        if not impl:
            return 0

        # pyrefly: ignore [not-iterable]
        module_path, replace_func = impl

        count = replace_methods(
            class_name="MoE",
            method_name="forward",
            new_method=replace_func,
            package=module_path,
        )

        return count
