# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn

from ..base_converter import BaseConverter
from ..convert_utils import replace_methods
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)


# ---- SFA (SparseAttention) ----


def npu_sparse_attn_shared_kv(
    query,
    ori_kv,
    cmp_kv,
    cmp_sparse_indices,
    sinks,
    softmax_scale,
    cmp_ratio,
    ori_mask_mode=4,
    cmp_mask_mode=3,
    ori_win_left=127,
    ori_win_right=0,
):
    cu_seq_lens_q = cu_seq_lens_ori_kv = cu_seq_lens_cmp_kv = None  # not support TND
    ori_sparse_indices = None  # ori kv use band mode
    batch_size, max_seq_len_q, num_heads_q, head_dim = query.size()
    num_heads_kv = 1
    max_seq_len_kv = ori_kv.size(1)
    topk = 0 if cmp_ratio != 4 else cmp_sparse_indices.size(-1)
    layout_q = layout_kv = "BSND"
    query = query.contiguous()  # [S, B, N, D] --> [B, S, N, D]
    ori_kv = ori_kv.unsqueeze(2).contiguous()  # [S, B, D] --> [B, S, 1, D]
    cmp_kv = (
        cmp_kv if cmp_kv is None else cmp_kv.unsqueeze(2).contiguous()
    )  # [S, B, D] --> [B, S, 1, D]
    if cmp_ratio != 4:
        cmp_sparse_indices = None
    else:
        cmp_sparse_indices = cmp_sparse_indices.unsqueeze(2).contiguous()
    # pyrefly: ignore [missing-import]
    from mindspeed.ops.npu_sparse_attn_shared_kv import SparseAttnSharedKV

    output = SparseAttnSharedKV.apply(
        query,
        ori_kv,
        cmp_kv,
        cu_seq_lens_q,
        cu_seq_lens_ori_kv,
        cu_seq_lens_cmp_kv,
        ori_sparse_indices,
        cmp_sparse_indices,
        sinks,
        softmax_scale,
        cmp_ratio,
        ori_mask_mode,
        cmp_mask_mode,
        ori_win_left,
        ori_win_right,
        num_heads_q,
        num_heads_kv,
        head_dim,
        batch_size,
        max_seq_len_q,
        max_seq_len_kv,
        topk,
        layout_q,
        layout_kv,
    )
    return output.contiguous()


def sdpa_to_sfa_adapter(
    self, query_states, kv_states, attn_sink, kv_compress, compress_topk_idxs
):

    if compress_topk_idxs is not None:
        if compress_topk_idxs.dtype != torch.int32:
            compress_topk_idxs = compress_topk_idxs.to(torch.int32)

    output = npu_sparse_attn_shared_kv(
        query=query_states,
        ori_kv=kv_states,
        cmp_kv=kv_compress,
        cmp_sparse_indices=compress_topk_idxs,
        sinks=attn_sink.float(),
        softmax_scale=self.softmax_scale,
        cmp_ratio=self.compress_ratio,
    )
    return output


# ---- LI (LiCompute) ----


def sdpa_to_li_adapter(
    self,
    q_indexer: torch.Tensor,
    k_indexer: torch.Tensor,
    weights: torch.Tensor,
    seqlen: int,
    offset: int,
):

    q_indexer = q_indexer.to(torch.bfloat16)
    k_indexer = k_indexer.to(torch.bfloat16).unsqueeze(2)
    weights = weights.to(torch.bfloat16)
    # pyrefly: ignore [missing-import]
    import mindspeed.ops.npu_lightning_indexer as mindspeed_li

    compress_topk_idxs, index_score = mindspeed_li.npu_lightning_indexer(
        q_indexer,
        k_indexer,
        weights,
        sparse_count=self.index_topk,
        sparse_mode=3,
        cmp_ratio=self.ratio,
    )

    compress_topk_idxs = compress_topk_idxs.squeeze(2)
    index_score = index_score.squeeze(2)
    if offset != 0:
        # pyrefly: ignore [no-matching-overload]
        compress_topk_idxs = torch.where(
            compress_topk_idxs == -1, compress_topk_idxs, compress_topk_idxs + offset
        )

    return compress_topk_idxs, index_score


# ---- LI Loss (LiLoss) ----

ms_npu_sparse_lightning_indexer_grad_kl_loss = None


def _get_ms_npu_sparse_lightning_indexer_grad_kl_loss():
    global ms_npu_sparse_lightning_indexer_grad_kl_loss
    if ms_npu_sparse_lightning_indexer_grad_kl_loss is None:
        # pyrefly: ignore [missing-import]
        from mindspeed.op_builder.npu_sparse_lightning_indexer_grad_kl_loss_builder import (
            NPUSparseLIGradKlLossOpBuilder,
        )

        ms_npu_sparse_lightning_indexer_grad_kl_loss = (
            NPUSparseLIGradKlLossOpBuilder()
            .load()
            .npu_sparse_lightning_indexer_grad_kl_loss
        )
    return ms_npu_sparse_lightning_indexer_grad_kl_loss


class SparseLightningIndexerGradKLLossWrapper(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(
        ctx,
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        loss_tracker,
        scale_value,
        cmp_ratio,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    ):
        ctx.save_for_backward(
            query, key, query_index, key_index, weights, sparse_indices
        )
        ctx.loss_tracker = loss_tracker
        ctx.scale_value = scale_value
        ctx.cmp_ratio = cmp_ratio
        ctx.actual_seq_qlen = actual_seq_qlen
        ctx.actual_seq_klen = actual_seq_klen
        ctx.layout = layout
        ctx.sparse_mode = sparse_mode
        ctx.pre_tokens = pre_tokens
        ctx.next_tokens = next_tokens

        # Return dummy loss during fwd, real operation will be postponed
        # to bwd, to avoid redundant computation of the loss function in
        # case where activation checkpointing is enabled.
        return torch.zeros(1, dtype=torch.float32, device=query.device)[0]

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        query, key, query_index, key_index, weights, sparse_indices = ctx.saved_tensors
        softmax_max = softmax_sum = query_rope = key_rope = None

        (
            d_query_index,
            d_key_index,
            d_weights,
            loss,
        ) = _get_ms_npu_sparse_lightning_indexer_grad_kl_loss()(
            query,
            key,
            query_index,
            key_index,
            weights,
            sparse_indices,
            softmax_max,
            softmax_sum,
            query_rope,
            key_rope,
            ctx.actual_seq_qlen,
            ctx.actual_seq_klen,
            ctx.scale_value,
            ctx.layout,
            ctx.sparse_mode,
            ctx.pre_tokens,
            ctx.next_tokens,
            ctx.cmp_ratio,
        )

        bsz, slen, *_ = query.shape
        token_scale = 1 / (bsz * slen)
        grad_scale = grad * token_scale

        d_query_index = d_query_index * grad_scale
        d_key_index = d_key_index * grad_scale
        d_weights = d_weights * grad_scale
        loss = loss * token_scale

        ctx.loss_tracker(loss[0])
        return None, None, d_query_index, d_key_index, d_weights, *([None] * 10)


# Wrapper for autograd.Function to support default/keyword argument
def npu_sparse_lightning_indexer_grad_kl_loss(
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    *,
    loss_tracker,
    scale_value,
    cmp_ratio,
    actual_seq_qlen=None,
    actual_seq_klen=None,
    layout="BSND",
    sparse_mode=3,
    pre_tokens=2147483647,
    next_tokens=2147483647,
):
    return SparseLightningIndexerGradKLLossWrapper.apply(
        query,
        key,
        query_index,
        key_index,
        weights,
        sparse_indices,
        loss_tracker,
        scale_value,
        cmp_ratio,
        actual_seq_qlen,
        actual_seq_klen,
        layout,
        sparse_mode,
        pre_tokens,
        next_tokens,
    )


def li_loss_adapter(
    self,
    query,
    key,
    query_index,
    key_index,
    weights,
    sparse_indices,
    indexer_score,
    attention_masks,
    offset,
):
    return npu_sparse_lightning_indexer_grad_kl_loss(
        query,
        key.unsqueeze(2),
        query_index,
        key_index.unsqueeze(2),
        weights,
        sparse_indices.unsqueeze(2),
        loss_tracker=self.save_loss,
        scale_value=self.softmax_scale,
        cmp_ratio=self.compress_ratio,
    )


# ---- Combined Converter ----


@register_npu_converter("deepseek_v4_sfa")
class DeepSeekV4SFAKernel(BaseConverter):

    MODEL_PACKAGE = "torchtitan_npu.models.deepseek_v4"
    SUPPORTED_MODELS = {"deepseek_v4"}

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> nn.Module:
        pkg = cls.MODEL_PACKAGE
        total = 0

        count = replace_methods(
            "SparseAttention", "forward", sdpa_to_sfa_adapter, package=pkg
        )
        logger.info(f"  [SparseAttention forward] Applied {count} replacement(s)")
        total += count

        count = replace_methods("LiCompute", "forward", sdpa_to_li_adapter, package=pkg)
        logger.info(f"  [LiCompute forward] Applied {count} replacement(s)")
        total += count

        count = replace_methods("LiLoss", "forward", li_loss_adapter, package=pkg)
        logger.info(f"  [LiLoss forward] Applied {count} replacement(s)")
        total += count

        # pyrefly: ignore [bad-return]
        return total
