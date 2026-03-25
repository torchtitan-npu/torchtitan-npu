# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torch.distributed.device_mesh import DeviceMesh
from torchtitan.models.attention import ScaledDotProductAttentionWrapper

from torchtitan_npu.patches.distributed.custom_context_parallel import (
    CustomContextParallelContext,
)


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


class UlyssesContextParallelContext(CustomContextParallelContext):

    """
    The base class already shards input tensors on sequence dimension when entering.
    This class performs ulysses CP by inserting 2 all-to-all operations to the input
    and output of the attention operation.

    1. Attention input:
        Perform all-to-all to scatter the head dimension and gather the sequence dimension.
        > Allows each NPU complete context access for each sequence, utilizing the stable
        head dimension to split the data, without restricting the choice of  local batch_size.

    2. Attention output:
        Perform all-to-all to scatter the sequence dimension and gather the head dimension.
        > Ensures compatibility with the rest of the transformer block that expects data
        to be partitioned across the sequence dimension.
    """

    def __init__(
        self,
        mesh: DeviceMesh,
        *,
        buffers: list[torch.Tensor] | None = None,
        buffer_seq_dims: list[int] | None = None,
        no_restore_buffers: set[torch.Tensor] | None = None,
        load_balance: bool = False,
    ):
        super().__init__(
            mesh,
            buffers=buffers,
            buffer_seq_dims=buffer_seq_dims,
            no_restore_buffers=no_restore_buffers,
            load_balance=load_balance,
        )
        self.cp_mesh = mesh
        self._orig_sdpa_forward = None

    @torch.no_grad()
    def __enter__(self):
        super().__enter__()  # This shards the buffers
        self.apply_patches()

    @torch.no_grad()
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original forward method."""
        if self._orig_sdpa_forward is not None:
            ScaledDotProductAttentionWrapper.forward = self._orig_sdpa_forward
        super().__exit__(exc_type, exc_val, exc_tb)

    def apply_patches(self):
        """Patch the attention forward method to use Ulysses-style context parallelism."""

        ScaledDotProductAttentionWrapper.cp_mesh = self.cp_mesh
        orig_sdpa_forward = (
            self._orig_sdpa_forward
        ) = ScaledDotProductAttentionWrapper.forward

        @functools.wraps(ScaledDotProductAttentionWrapper.forward)
        def patched_forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            *,
            scale: float | None = None,
        ):
            """
            Forward pass with Ulysses-style context parallelism.

            Input layout: [batch_size, n_heads, seq_len // CP, head_dim]
            All2all for attention input: [batch_size, n_heads // CP, seq_len, head_dim]
            After attention operation: [batch_size, n_heads // CP, seq_len, head_dim]
            All2all for attention output: [batch_size, n_heads, seq_len // CP, head_dim]
            """
            q = all_to_all(q, self.cp_mesh, scatter_dim=1, gather_dim=2)
            k = all_to_all(k, self.cp_mesh, scatter_dim=1, gather_dim=2)
            v = all_to_all(v, self.cp_mesh, scatter_dim=1, gather_dim=2)

            output = orig_sdpa_forward(self, q, k, v, scale=scale)

            # Reverse the redistribution to return the output to the original sequence-parallel layout.
            output = all_to_all(output, self.cp_mesh, scatter_dim=2, gather_dim=1)

            return output

        ScaledDotProductAttentionWrapper.forward = patched_forward
