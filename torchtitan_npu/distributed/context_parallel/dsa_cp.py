# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.distributed as dist
import torch_npu
from torch.distributed._tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.device_mesh import DeviceMesh

from torchtitan_npu.converters.kernels.dsa import SparseLightningIndexerKLLoss
from torchtitan_npu.models.deepseek_v32.model.model import (
    DSASparseAttention,
    DSV32_SDPA,
)


INDEXER_GRAD_INDICES = [2, 3, 4]  # corresponding to LILossTrain.backward outputs
USE_DTENSOR_MODE = True  # whether to use dtensor to communicate


class AllgatherOnSequence(torch.autograd.Function):
    """Allgather with backward on sequence dim of BSND tensor."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, tensor, mesh):
        group = mesh.get_group()
        cp_size = dist.get_world_size(group=group)

        b, s_local, n, d = tensor.shape
        s_global = s_local * cp_size
        # BSND -> SBND
        tensor_permuted = tensor.permute(1, 0, 2, 3).contiguous()
        # AllGather tensor in given group on sequence dim
        output_buffer = torch.empty(
            (s_global, b, n, d), dtype=tensor.dtype, device=tensor.device
        )
        dist.all_gather_into_tensor(output_buffer, tensor_permuted, group=group)
        # SBND -> BSND
        tensor_global = output_buffer.permute(1, 0, 2, 3)

        ctx.group = group
        ctx.s_global = s_global
        ctx.orig_shape = tensor.shape
        return tensor_global

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None

        group = ctx.group
        b, s_local, n, d = ctx.orig_shape
        # BSND -> SBND
        grad_output_permuted = grad_output.permute(1, 0, 2, 3).contiguous()
        # ReduceScatter grad: input [s_global, b, n, d] -> output [s_local, b, n, d]
        grad_local_permuted = torch.empty(
            (s_local, b, n, d), dtype=grad_output.dtype, device=grad_output.device
        )
        dist.reduce_scatter_tensor(
            grad_local_permuted, grad_output_permuted, group=group
        )
        # SBND -> BSND
        grad_input = grad_local_permuted.permute(1, 0, 2, 3)
        return grad_input, None


class ToLocalWithPartialGrad(torch.autograd.Function):
    """DTensor to local with reduce-scatter as the backward (via Partial placement)."""

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, dtensor, mesh):
        ctx.mesh = mesh
        return dtensor.to_local()

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None
        grad_dtensor = DTensor.from_local(grad_output, ctx.mesh, placements=[Partial()])
        return grad_dtensor, None


def dtensor_allgather_sequence(tensor, mesh):
    """Allgather a sequence-sharded local tensor via DTensor; backward is reduce-scatter."""
    dtensor = DTensor.from_local(tensor, mesh, placements=[Shard(1)])
    dtensor_global = dtensor.redistribute(mesh, placements=[Replicate()])
    return ToLocalWithPartialGrad.apply(dtensor_global, mesh)


def allgather_sequence(tensor, mesh):
    if USE_DTENSOR_MODE:
        return dtensor_allgather_sequence(tensor, mesh)
    else:
        return AllgatherOnSequence.apply(tensor, mesh)


def _allgather_on_dim(tensor: torch.Tensor, tp_mesh, dim: int) -> torch.Tensor:
    """All-gather a TP-sharded plain tensor along `dim` across TP ranks.

    Used in dsa_forward_with_cp to restore the full head count before calling
    npu_sparse_lightning_indexer_grad_kl_loss, which requires N ∈ {64, 128}.
    No gradient support is needed: LILossTrain.backward returns None for
    query and softmax_max/sum, so no gradient flows through these tensors.
    """
    group = tp_mesh.get_group()
    tp_size = dist.get_world_size(group=group)
    if tp_size == 1:
        return tensor
    ndim = tensor.ndim
    # Permute so the sharded dim is at position 0 for all_gather_into_tensor
    perm = [dim] + [i for i in range(ndim) if i != dim]
    inv_perm = [0] * ndim
    for dst, src in enumerate(perm):
        inv_perm[src] = dst
    t = tensor.permute(perm).contiguous()
    out_shape = list(t.shape)
    out_shape[0] *= tp_size
    output = torch.empty(out_shape, dtype=t.dtype, device=t.device)
    dist.all_gather_into_tensor(output, t, group=group)
    return output.permute(inv_perm).contiguous()


def dsa_forward_with_cp(
    self,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    q_indexer: torch.Tensor | None = None,
    k_indexer: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    end_pos: torch.Tensor | None = None,
    index_topk: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Forward pass of the dsa module with context parallel.
    CP strategy: allgather KV tensors across the CP mesh.
    """

    if k.shape[1] != 1 or v.shape[1] != 1:
        raise NotImplementedError(
            "Only support num_head_kv == 1 in dsa forward under absorb mode."
        )

    # Gather full k_indexer and slice causally up to current rank
    k_indexer_global = allgather_sequence(k_indexer, self.cp_mesh)
    cp_rank = dist.get_rank(group=self.cp_mesh.get_group())
    s_local = k_indexer.shape[1]  # pyrefly: ignore [missing-attribute]
    slice_end = s_local * (cp_rank + 1)
    k_indexer_causal = k_indexer_global[:, :slice_end, :, :]

    # NOTE: set return_value=False to avoid torch.compile / DTensor meta path failure
    ret = torch_npu.npu_lightning_indexer(
        q_indexer,
        k_indexer_causal,
        weights,
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        layout_query="BSND",
        layout_key="BSND",
        sparse_count=index_topk,
        sparse_mode=3,
        return_value=False,
    )
    topk_indices = ret[0] if isinstance(ret, tuple) else ret

    # BNSD -> BSND
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    q_nope, q_pe = torch.split(
        q, [self.model_args.kv_lora_rank, self.model_args.qk_rope_head_dim], dim=-1
    )
    k_nope, k_pe = torch.split(
        k, [self.model_args.kv_lora_rank, self.model_args.qk_rope_head_dim], dim=-1
    )

    bsz = q.shape[0]
    actual_seq_len = torch.full(
        (bsz,), q_nope.shape[1], dtype=torch.int32, device=q_nope.device
    )
    actual_seq_len_kv = torch.full(
        (bsz,), slice_end, dtype=torch.int32, device=q_nope.device
    )

    # Allgather KV across CP ranks
    k_nope_global = allgather_sequence(k_nope, self.cp_mesh)
    v_global = allgather_sequence(v, self.cp_mesh)
    k_pe_global = allgather_sequence(k_pe, self.cp_mesh)

    output, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
        q_nope,
        k_nope_global[:, :slice_end, :, :],
        v_global[:, :slice_end, :, :],
        sparse_indices=topk_indices.to(torch.int32),
        block_table=None,
        actual_seq_lengths_query=actual_seq_len,
        actual_seq_lengths_kv=actual_seq_len_kv,
        query_rope=q_pe,
        key_rope=k_pe_global[:, :slice_end, :, :],
        scale_value=scale,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        attention_mode=2,  # 0: GQA/MHA, 1: MLA-naive, 2: MLA-absorb
        return_softmax_lse=True,  # must be True in training mode
    )

    tp_mesh = getattr(self, "tp_mesh", None)
    if tp_mesh is not None:
        q_nope = _allgather_on_dim(q_nope, tp_mesh, dim=2)
        q_pe = _allgather_on_dim(q_pe, tp_mesh, dim=2)
        softmax_max = _allgather_on_dim(softmax_max, tp_mesh, dim=3)
        softmax_sum = _allgather_on_dim(softmax_sum, tp_mesh, dim=3)

    loss = self.compute_dsa_indexer_loss(
        q_nope,
        k_nope_global[:, :slice_end, :, :],
        q_indexer,
        k_indexer_causal,
        weights,
        topk_indices,
        softmax_max,
        softmax_sum,
        scale_value=scale,
        query_rope=q_pe,
        key_rope=k_pe_global[:, :slice_end, :, :],
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout="BSND",
        sparse_mode=3,  # mask is rightDownCausal mode
    )
    output = output.transpose(1, 2)
    # pyrefly: ignore [bad-return]
    return loss, output


def patch_dsa_for_context_parallel(
    *,
    cp_mesh: DeviceMesh,
    model_args: object | None = None,
    tp_mesh: DeviceMesh | None = None,
) -> None:
    """
    Patch DSA attention forward to a CP-aware implementation.

    Called from the patched `apply_cp_to_attention_module(...)` when attention_type == "dsa".
    CP sharding (sequence dimension) is handled by TorchTitan v0.2.2 module-level CP.
    """
    for cls in (DSASparseAttention, DSV32_SDPA):
        cls.cp_mesh = cp_mesh
        if model_args is not None:
            cls.model_args = model_args  # pyrefly: ignore [no-access]
        cls.tp_mesh = tp_mesh
        cls.compute_dsa_indexer_loss = (  # pyrefly: ignore [no-access]
            SparseLightningIndexerKLLoss()
        )
        cls.forward = dsa_forward_with_cp  # pyrefly: ignore [bad-assignment]
