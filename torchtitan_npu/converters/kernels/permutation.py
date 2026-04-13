# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""MoE Unpermute API"""
import torch
import torch_npu


class NPUMoeTokenUnpermute(torch.autograd.Function):
    """functional npu_moe_token_unpermute"""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        permuted_tokens: torch.Tensor,
        sorted_indices: torch.Tensor,
        restore_shape: torch.Size,
    ) -> torch.Tensor:
        if not permuted_tokens.numel():
            return permuted_tokens

        output, _, _, _ = torch_npu._npu_moe_token_unpermute_with_routing_map(
            permuted_tokens,
            sorted_indices,
            restore_shape,
            probs=None,
            routing_map=None,
            drop_and_pad=False,
        )
        ctx.restore_shape = restore_shape
        ctx.sorted_indices = sorted_indices

        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, unpermuted_tokens_grad):
        if not unpermuted_tokens_grad.numel():
            return unpermuted_tokens_grad, None, None, None, None, None
        if ctx.needs_input_grad[0]:
            sorted_indices = ctx.sorted_indices
            act_grad, _ = torch_npu.npu_moe_token_unpermute_with_routing_map_grad(
                unpermuted_tokens_grad,
                sorted_indices,
                sorted_indices,
                routing_map=None,
                permuted_tokens=None,
                probs=None,
                drop_and_pad=False,
                restore_shape=ctx.restore_shape,
            )
            return act_grad, None, None, None, None, None

        return None, None, None, None, None, None
