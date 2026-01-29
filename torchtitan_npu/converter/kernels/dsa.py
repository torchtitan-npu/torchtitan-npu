# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Tuple
import torch
import torch_npu
import torch.nn as nn

from ..registry import (
    BaseKernel,
    KernelType,
    replace_methods,
)

logger = logging.getLogger(__name__)


class SparseLightningIndexerKLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        query,
        key,
        query_indexer,
        key_indexer,
        weights,
        topk_indices,
        softmax_max,
        softmax_sum,
        scale_value=1,
        *,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout='BSND',
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    ):
        """NPU Sparse Lightning Indexer KL Divergence Loss Function"""
        sq = query.shape[1]
        loss = LILossTrain.apply(query, key, query_indexer, key_indexer, weights, topk_indices, softmax_max,
                                 softmax_sum, scale_value, query_rope, key_rope, actual_seq_qlen, actual_seq_klen,
                                 layout, sparse_mode, pre_tokens, next_tokens, )
        return loss / sq


class LILossTrain(torch.autograd.Function):
    """
    A custom autograd function that computes kl loss in sparse lightning indexer.

    This interface implements the backward functionality of npu_lightning_indexer and integrates the loss computation.
    The npu_lightning_indexer selects the top-k pairs between queries and keys in attention that exhibit the strongest
    intrinsic correlations, storing them in sparse_indices. This reduces the computational cost of attention in
    long-sequence scenarios and improves training performance.
    """

    @staticmethod
    def forward(
        ctx,
        query,
        key,
        query_indexer,
        key_indexer,
        weights,
        sparse_indices,
        softmax_max,
        softmax_sum,
        scale_value=1,
        query_rope=None,
        key_rope=None,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout='BSND',
        sparse_mode=3,
        pre_tokens=65536,
        next_tokens=65536,
    ):
        """
        Forward pass: compute the total loss by processing hidden states in chunks.

        Args:
            ctx: Context object used to save tensors for backward pass.
            query (Tensor): Required. Represents the Attention query. Shapes: (B, S1, N1, D), (T1, N1, D)
            key (Tensor): Required. Represents the Attention key. Shapes: (B, S2, N2, D), (T2, N2, D)
            query_indexer (Tensor): Required. Input query for the lightning_indexer forward pass.
            key_indexer (Tensor): Required. Input key for the lightning_indexer forward pass.
            weights (Tensor): Required. Weight coefficients of lightning_indexer.
            sparse_indices (Tensor): Required. Token indices of sorted key and key_index.
            softmax_max (Tensor): Required. Maximum values from Attention softmax results.
            softmax_sum (Tensor): Required. Sum values from Attention softmax results.
            scale_value (float): Required scaling coefficient.
            query_rope (Tensor, optional): RoPE information for query in MLA architecture.
            key_rope (Tensor, optional): RoPE information for key in MLA architecture.
            actual_seq_qlen (list[int], optional): Required in TND layout. Cumulative sequence lengths for query.
            actual_seq_klen (list[int], optional): Required in TND layout. Cumulative sequence lengths for key.
            layout (str, optional): Input data layout format. Supported: "BSND", "TND". Default: "BSND".
            sparse_mode (int, optional): Sparse computation mode. Default: 3.
            pre_tokens (int, optional): Number of preceding tokens for sparse Attention. Default: 65536.
            next_tokens (int, optional): Number of succeeding tokens for sparse Attention. Default: 65536.
        Returns:
            d_query_index (Tensor): Gradient of query_index.
            d_key_index (Tensor): Gradient of key_index.
            d_weights (Tensor): Gradient of weights.
            loss (Tensor): Difference between network forward output and golden value.
        """

        d_query_index, d_key_index, d_weights, loss = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
            query,
            key,
            query_indexer,
            key_indexer,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            scale_value=scale_value,
            query_rope=query_rope,
            key_rope=key_rope,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout=layout,
            sparse_mode=sparse_mode,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
        )

        # Save computed gradients for use in backward pass
        ctx.save_for_backward(d_query_index, d_key_index, d_weights)
        return loss[0]

    @staticmethod
    def backward(ctx, *grad_output) -> Tuple:
        """
        Backward pass: propagate upstream gradients through the precomputed gradients.

        Args:
            ctx: Context object with saved tensors from forward pass.
            grad_output: Gradient output.

        Returns:
            tuple: Gradients.
        """
        d_query_index, d_key_index, d_weights = ctx.saved_tensors
        grad_scale = grad_output[0]
        if torch.ne(grad_scale, torch.tensor(1.0, device=grad_scale.device)):
            d_query_index = d_query_index * grad_scale
            d_key_index = d_key_index * grad_scale
            d_weights = d_weights * grad_scale

        res_list = [None] * 12
        return None, None, d_query_index, d_key_index, d_weights, *res_list


def dsa_forward(
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
    Forward pass of the dsa module.
    """
    if k.shape[1] != 1 or v.shape[1] != 1:
        raise NotImplementedError("Only support num_head_kv == 1 in dsa forward under absorb mode.")

    # Fuse LILossTrain includes LIG
    topk_indices, _ = torch_npu.npu_lightning_indexer(
        q_indexer,
        k_indexer,
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

    output, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
        q_nope, k_nope, v,
        sparse_indices=topk_indices.to(torch.int32),
        block_table=None,
        actual_seq_lengths_query=actual_seq_len,
        actual_seq_lengths_kv=actual_seq_len,
        query_rope=q_pe,
        key_rope=k_pe,
        scale_value=scale,
        sparse_block_size=1,
        layout_query='BSND',
        layout_kv='BSND',
        sparse_mode=3,
        attention_mode=2,            # 0: GQA/MHA, 1: MLA-naive, 2: MLA-absorb
        return_softmax_lse=True,     # it must be True in training mode
    )

    # The loss is actually computed by SparseLightningIndexerKLLoss.forward
    # If tp is enabled, inner_attention.compute_dsa_indexer_loss is patched in deepseek_v32_parallelize.py
    # Otherwise, inner_attention.compute_dsa_indexer_loss is patched in this file
    loss = self.compute_dsa_indexer_loss(
        q_nope,
        k_nope,
        q_indexer,
        k_indexer,
        weights,
        topk_indices,
        softmax_max,
        softmax_sum,
        scale_value=scale,
        query_rope=q_pe,
        key_rope=k_pe,
        actual_seq_qlen=None,
        actual_seq_klen=None,
        layout='BSND',
    )
    output = output.transpose(1, 2)
    return loss, output


class DSAKernel(BaseKernel):

    kernel_type = KernelType.DSA
    MODEL_PACKAGE = "torchtitan_npu.models.deepseek_v32"

    @classmethod
    def apply(cls, model: nn.Module, **kwargs) -> nn.Module:
        pkg = cls.MODEL_PACKAGE

        count = replace_methods("DSV32_SDPA", "forward", dsa_forward, package=pkg)
        logger.info(f"  [DSV32_SDPA forward] Applied {count} replacement(s)")
        logger.info(f"  Only matrix absorb mode is supported, and LI Loss is enabled by default.")

        # If tp is no enabled, then the indexer_loss patch in deepseek_v32_parallelize.py won't be applied
        # The patch is applied here as a supplement
        for transformer_block in model.layers.values():
            inner_attention = transformer_block.attention.inner_attention
            if not isinstance(inner_attention.compute_dsa_indexer_loss, SparseLightningIndexerKLLoss):
                inner_attention.compute_dsa_indexer_loss = SparseLightningIndexerKLLoss()
        return model
