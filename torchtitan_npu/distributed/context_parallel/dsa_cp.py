# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
import torch.distributed as dist
from torch.distributed._tensor import DTensor, Replicate, Shard, Partial
import torch_npu

from torchtitan_npu.converter.kernels.dsa import SparseLightningIndexerKLLoss, LILossTrain
from torchtitan_npu.models.deepseek_v32.model.model import DSV32_SDPA
from torchtitan_npu.patches.distributed.custom_context_parallel import CustomContextParallelContext


INDEXER_GRAD_INDICES = [2, 3, 4]    # corresponding to LILossTrain.backward outputs
USE_DTENSOR_MODE = True             # whether use dtensor to communicate


class AllgatherOnSequence(torch.autograd.Function):
    """Allgather with backward on sequence dim of BSND tensor."""

    @staticmethod
    def forward(ctx, tensor, mesh):
        group = mesh.get_group()
        cp_rank = dist.get_rank(group=group)
        cp_size = dist.get_world_size(group=group)

        b, s_local, n, d = tensor.shape
        s_global = s_local * cp_size
        # BSND -> SBND
        tensor_permuted = tensor.permute(1, 0, 2, 3).contiguous()
        # AllGather tensor in given group on sequence dim
        output_buffer = torch.empty((s_global, b, n, d), dtype=tensor.dtype, device=tensor.device)
        dist.all_gather_into_tensor(output_buffer, tensor_permuted, group=group)
        # SBND -> BSND
        tensor_global = output_buffer.permute(1, 0, 2, 3)

        ctx.group = group
        ctx.s_global = s_global
        ctx.orig_shape = tensor.shape
        return tensor_global

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None

        group = ctx.group
        s_global = ctx.s_global
        b, s_local, n, d = ctx.orig_shape
        # BSND -> SBND
        grad_output_permuted = grad_output.permute(1, 0, 2, 3).contiguous()
        # ReduceScatter grad: input [s_global, b, n, d] -> output [s_local, b, n, d]
        grad_local_permuted = torch.empty((s_local, b, n, d), dtype=grad_output.dtype, device=grad_output.device)
        dist.reduce_scatter_tensor(grad_local_permuted, grad_output_permuted, group=group)
        # SBND -> BSND
        grad_input = grad_local_permuted.permute(1, 0, 2, 3)
        return grad_input, None


class ToLocalWithPartialGrad(torch.autograd.Function):
    """DTensor to local with specified partial grad from backward."""

    @staticmethod
    def forward(ctx, dtensor, mesh):
        ctx.mesh = mesh
        return dtensor.to_local()

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None

        grad_dtensor = DTensor.from_local(grad_output, ctx.mesh, placements=[Partial()])
        return grad_dtensor, None


def dtensor_allgather_sequence(local_tensor, mesh):
    # local_tensor -> DTensor
    dtensor = DTensor.from_local(local_tensor, mesh, placements=[Shard(1)])
    # allgather DTensor
    dtensor_global = dtensor.redistribute(mesh, placements=[Replicate()])
    # return normal tensor, ToLocalWithPartialGrad is used to ensure the backward of allgather is reducescatter
    return ToLocalWithPartialGrad.apply(dtensor_global, mesh)


def allgather_sequence(tensor, mesh):
    if USE_DTENSOR_MODE:
        return dtensor_allgather_sequence(tensor, mesh)
    else:
        return AllgatherOnSequence.apply(tensor, mesh)


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
    Forward pass of the dsa module with context parallel, based on torchtitan_npu.converter.kernels.dsa.dsa_forward
    Out context parallel strategy for DSA is to simply allgather KV tensors because these tensors are relatively small
    """
    if k.shape[1] != 1 or v.shape[1] != 1:
        raise NotImplementedError("Only support num_head_kv == 1 in dsa forward under absorb mode.")

    # get complete k_indexer in cp_group by allgather
    k_indexer_global = allgather_sequence(k_indexer, self.cp_mesh)

    # to ensure causal attn in npu_lightning_indexer, should slice the key tensor
    cp_rank = dist.get_rank(group=self.cp_mesh.get_group())
    s_local = k_indexer.shape[1]
    slice_end = s_local * (cp_rank + 1)

    topk_indices, _ = torch_npu.npu_lightning_indexer(
        q_indexer,
        k_indexer_global[:, : slice_end, :, :],
        weights,
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        layout_query='BSND',
        layout_key='BSND',
        sparse_count=index_topk,
        sparse_mode=3,
        return_value=True,
    )

    # To BSND
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Split q_nope / q_pe
    q_nope, q_pe = torch.split(q, [self.model_args.kv_lora_rank, self.model_args.qk_rope_head_dim], dim=-1)
    k_nope, k_pe = torch.split(k, [self.model_args.kv_lora_rank, self.model_args.qk_rope_head_dim], dim=-1)

    bsz = q.shape[0]
    actual_seq_len = torch.full((bsz,), q_nope.shape[1], dtype=torch.int32, device=q_nope.device)

    # the causal attn can be ensured by topk_indices in npu_sparse_flash_attention, do not need to slice key tensors
    k_nope_global = allgather_sequence(k_nope, self.cp_mesh)
    v_global = allgather_sequence(v, self.cp_mesh)
    k_pe_global = allgather_sequence(k_pe, self.cp_mesh)

    output, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
        q_nope, k_nope_global, v_global,
        sparse_indices=topk_indices.to(torch.int32),
        block_table=None,
        actual_seq_lengths_query=actual_seq_len,
        actual_seq_lengths_kv=actual_seq_len * self.cp_mesh.size(),
        query_rope=q_pe,
        key_rope=k_pe_global,
        scale_value=scale,
        sparse_block_size=1,
        layout_query='BSND',
        layout_kv='BSND',
        sparse_mode=3,
        attention_mode=2,            # 0: GQA/MHA, 1: MLA-naive, 2: MLA-absorb
        return_softmax_lse=True,     # it must be True in training mode
    )

    # to ensure causal attn in npu_sparse_lightning_indexer_grad_kl_loss, should slice the key tensors
    loss = self.compute_dsa_indexer_loss(
        q_nope,
        k_nope_global[:, : slice_end, :, :],
        q_indexer,
        k_indexer_global[:, : slice_end, :, :],
        weights,
        topk_indices,
        softmax_max,
        softmax_sum,
        scale_value=scale,
        query_rope=q_pe,
        key_rope=k_pe_global[:, : slice_end, :, :],
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout='BSND',
        sparse_mode=3   # mask is rightDownCausal mode
    )
    output = output.transpose(1, 2)

    # indexer loss average, this only affects the loss display
    dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.cp_mesh.get_group())
    loss /= self.cp_mesh.size()
    return loss, output


def patch_indexer_grad(scale_factor):
    if not hasattr(LILossTrain, '_original_backward_backup'):
        LILossTrain._original_backward_backup = LILossTrain.backward
    original_bwd = LILossTrain._original_backward_backup

    @functools.wraps(original_bwd)
    def wrapped_backward(ctx, *grad_outputs):
        grads = original_bwd(ctx, *grad_outputs)
        grads_list = list(grads)
        for idx in INDEXER_GRAD_INDICES:
            grads_list[idx] *= scale_factor     # scale the output grads
        return tuple(grads_list)

    LILossTrain.backward = staticmethod(wrapped_backward)


def patch_dsa_forward_check():
    @functools.wraps(dsa_forward_with_cp)
    def _forward_wrapper(self, *args, **kwargs):
        # check if need to patch the indexer loss computation module
        current_loss_impl = getattr(self, "compute_dsa_indexer_loss", None)
        if not isinstance(current_loss_impl, SparseLightningIndexerKLLoss):
            self.compute_dsa_indexer_loss = SparseLightningIndexerKLLoss()

        return dsa_forward_with_cp(self, *args, **kwargs)

    # patch the forward with dsa cp version
    DSV32_SDPA.forward = _forward_wrapper


class AscendDSAContextParallelContext(CustomContextParallelContext):
    """Context parallel context manager for DSA on Ascend NPU hardware."""

    @torch.no_grad()
    def __enter__(self):
        self.apply_patches()
        super().__enter__()

    def apply_patches(self):
        DSV32_SDPA.cp_mesh = self.mesh
        # inner_attention forward will be patched with the CP version
        patch_dsa_forward_check()

        # scale the grad to ensure the accuracy
        cp_size = self.mesh.size()
        patch_indexer_grad(scale_factor=cp_size)