# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.models.attention import ScaledDotProductAttentionWrapper


class AllToAll(torch.autograd.Function):
    """
    All-to-all operation with proper gradient handling.
    Performs all-to-all in forward and backward passes.
    """

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, input_tensor, mesh, scatter_dim, gather_dim):
        ctx.mesh = mesh
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        world_size = mesh.size()

        # Split along scatter_dim
        input_list = [
            t.contiguous()
            for t in list(input_tensor.chunk(world_size, dim=scatter_dim))
        ]
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

        # All-to-all
        torch.distributed.all_to_all(output_list, input_list, group=mesh.get_group())
        # Concatenate along gather_dim
        output = torch.cat(output_list, dim=gather_dim)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # All-to-all in reverse: swap scatter and gather dims
        world_size = ctx.mesh.size()

        # Backward of "Concatenate along gather_dim" - split along forward's gather_dim
        grad_list = [
            t.contiguous()
            for t in list(grad_output.chunk(world_size, dim=ctx.gather_dim))
        ]
        grad_output_list = [torch.empty_like(grad_list[0]) for _ in range(world_size)]

        # Reversed all-to-all
        torch.distributed.all_to_all(
            grad_output_list, grad_list, group=ctx.mesh.get_group()
        )

        # Backward of "Split along scatter_dim" - concatenate along forward's scatter_dim
        grad_input = torch.cat(grad_output_list, dim=ctx.scatter_dim)

        return grad_input, None, None, None


def all_to_all(input_tensor, mesh, scatter_dim, gather_dim):
    """
    Wrapper for all-to-all operation.

    Args:
        input_tensor: Input tensor
        mesh: Device mesh for CP group
        scatter_dim: Dimension to scatter (split and distribute)
        gather_dim: Dimension to gather (collect and concatenate)
    """
    return AllToAll.apply(input_tensor, mesh, scatter_dim, gather_dim)


def patch_ulysses_for_context_parallel(*, cp_mesh: DeviceMesh) -> None:
    """
    Patch ScaledDotProductAttentionWrapper.forward to a Ulysses CP-aware implementation.

    Called from apply_cp_to_attention_module when attention_type == "ulysses".
    Replaces the class-level forward so every instance automatically performs
    AllToAll before and after the SDPA kernel.

    Input layout:  [B, n_heads,      seq // CP, head_dim]
    After A2A:     [B, n_heads // CP, seq,       head_dim]
    After SDPA:    [B, n_heads // CP, seq,       v_head_dim]
    After A2A:     [B, n_heads,       seq // CP, v_head_dim]
    """
    ScaledDotProductAttentionWrapper.cp_mesh = cp_mesh
    orig_forward = ScaledDotProductAttentionWrapper.forward

    @functools.wraps(orig_forward)
    def patched_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        scale: float | None = None,
    ):
        q = all_to_all(q, self.cp_mesh, scatter_dim=1, gather_dim=2)
        k = all_to_all(k, self.cp_mesh, scatter_dim=1, gather_dim=2)
        v = all_to_all(v, self.cp_mesh, scatter_dim=1, gather_dim=2)
        output = orig_forward(self, q, k, v, scale=scale)
        return all_to_all(output, self.cp_mesh, scatter_dim=2, gather_dim=1)

    ScaledDotProductAttentionWrapper.forward = (
        patched_forward  # pyrefly: ignore [bad-assignment]
    )
