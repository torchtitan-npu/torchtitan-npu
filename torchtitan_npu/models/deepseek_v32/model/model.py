# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from typing import ClassVar

import torch
import torch.nn.functional as F
from einops import rearrange

# pyrefly: ignore [missing-import]
from scipy.linalg import hadamard
from torch import nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from torchtitan.models.deepseek_v3.model.model import (
    DeepSeekV3Model,
    precompute_freqs_cis,
    TransformerBlock,
)
from torchtitan.protocols.model import AttentionMasksType

from torchtitan_npu.converters.kernels.rope_broadcast import reshape_for_broadcast
from .args import DeepSeekV32ModelArgs

logger = logging.getLogger()


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
    interleaved: bool = True,
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.
    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.
        interleaved: If False, use non-interleaved layout (Indexer path).
        positions: Global position ids when sequence is sharded (e.g. CP); optional.
    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """
    dtype = x.dtype
    shape = x.shape
    if not interleaved:
        x = x.view(*shape[:-1], 2, -1).transpose(-1, -2).contiguous()
    x = torch.view_as_complex(x.float().view(*shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, x, positions)
    y = torch.view_as_real(x * freqs_cis).flatten(3)
    if not interleaved:
        y = torch.cat([y[..., 0::2], y[..., 1::2]], dim=-1)
    return y.to(dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)

    def reset_parameters(self):
        if self.weight is not None:
            nn.init.ones_(self.weight)


def hadamard_transform_ref(x, scale=1.0):
    """
    Eager implementation of the Hadamard transform
    Args:
        x:(torch.Tensor): input tensor
    """

    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        # pyrefly: ignore [bad-argument-type]
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(
        x,
        torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device),
    )
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:

    hadamard_transform = hadamard_transform_ref
    hidden_size = x.size(-1)
    return hadamard_transform(x, scale=hidden_size**-0.5)


def bf16_index(q: torch.Tensor, weight: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Perform index score using BF16 precision.
    """
    query = rearrange(q, "b s h d -> b h s d").to(torch.float32)
    key = rearrange(k, "b s h d -> b h d s").to(torch.float32)
    p = torch.matmul(query, key)
    relu_out = torch.nn.functional.relu(p)

    weight_out = relu_out * weight.permute(0, 2, 1, 3)

    reduce_out = torch.sum(weight_out, dim=1)
    return reduce_out


class Indexer(torch.nn.Module):
    def __init__(self, args: DeepSeekV32ModelArgs):
        super().__init__()
        self.dim: int = args.dim
        self.n_heads: int = args.index_n_heads
        self.head_dim: int = args.index_head_dim
        self.rope_head_dim: int = args.qk_rope_head_dim
        self.index_topk: int = args.index_topk
        self.q_lora_rank: int = args.q_lora_rank
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.head_dim, bias=False
        )
        self.wk = nn.Linear(self.dim, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim)
        # weights_proj in the checkpoint is stored in bf16, while the parameters here are stored in fp32 for convenient.
        self.weights_proj = nn.Linear(
            self.dim, self.n_heads, dtype=torch.float32, bias=False
        )
        self.softmax_scale = self.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,
        qr: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None,
        positions: torch.Tensor | None = None,
    ):
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen
        q = self.wq_b(qr)
        q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(
            q, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        # rope in indexer is not interleaved
        q_pe = apply_rotary_emb(q_pe, freqs_cis, False, positions)
        q = torch.cat([q_pe, q_nope], dim=-1)
        k = self.wk(x)
        k = self.k_norm(k)
        k_pe, k_nope = torch.split(
            k, [self.rope_head_dim, self.head_dim - self.rope_head_dim], dim=-1
        )
        # rope in indexer is not interleaved
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis, False, positions).squeeze(
            2
        )
        k = torch.cat([k_pe, k_nope], dim=-1).unsqueeze(2)
        q = rotate_activation(q)
        k = rotate_activation(k)
        weights = self.weights_proj(x) * self.n_heads**-0.5
        weights = weights * self.softmax_scale
        return q, weights, k, end_pos

    def init_weights(self, init_std: float):
        linear_list = [self.wq_b, self.wk, self.weights_proj]
        linear_list.extend([self.wq_b, self.wk, self.weights_proj])
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        self.k_norm.reset_parameters()


LOSS_SCALE = torch.tensor(1.0)


class DSAIndexerLossAutoScaler(torch.autograd.Function):
    """An AutoScaler that triggers the backward pass and scales the grad for DSA indexer loss."""

    # pyrefly: ignore [bad-assignment]
    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Preserve the indexer_loss by storing it in the context to avoid garbage collection.

        Args:
            output (torch.Tensor): The output tensor.
            aux_loss (torch.Tensor): The indexer loss tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output: torch.Tensor):
        """Compute and scale the gradient for indexer loss.

        Args:
            grad_output (torch.Tensor): The gradient of the output.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The gradient of the output, scaled indexer loss
                                               gradient.
        """
        (loss,) = ctx.saved_tensors
        LOSS_SCALE.to(device=loss.device)
        scaled_dsa_indexer_loss_grad = torch.ones_like(loss) * LOSS_SCALE
        return grad_output, scaled_dsa_indexer_loss_grad

    @classmethod
    def set_loss_scale(cls, scale: torch.Tensor) -> None:
        global LOSS_SCALE
        LOSS_SCALE = scale


class DSAIndexerLossLoggingHelper:
    """Helper class for logging DSAIndexer losses."""

    tracker = {}

    @staticmethod
    def save_loss_to_tracker(
        loss: torch.Tensor,
        layer_number: int,
        num_layers: int,
    ):
        """Save the DSA indexer loss for logging.
        Args:
            loss (torch.Tensor): The loss tensor.
            layer_number (int): Layer index of the loss.
            num_layers (int): The number of total layers.
        """
        # Skip DSA indexer loss logging if layer_number is None.
        if layer_number is None:
            return

        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=loss.device)
        loss_val = loss.detach()
        if hasattr(loss_val, "to_local"):
            loss_val = loss_val.to_local()
        tracker["values"][layer_number - 1] += loss_val

    @staticmethod
    def clean_loss_in_tracker():
        """Clear the DSA indexer losses."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        tracker["values"].zero_()

    @staticmethod
    def track_dsa_indexer_metrics(total_acc_steps: int):
        """Track the DSA Indexer metrics for logging."""
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return
        das_indexer_losses = tracker["values"]
        das_indexer_num_layers = das_indexer_losses.shape[0]
        loss = das_indexer_losses.sum() / das_indexer_num_layers / total_acc_steps
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        logger.info(f"indexer loss: {loss.item()}")


class DSAIndexerLoss(torch.nn.Module):
    """Compute dsa indexer loss at sparse training stage
    Reference: https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf
    Args:
        main_attn_dist: Q dist
        index_score: P dist
        topk_indices: Selected top-K indices for sparse phase
        loss_scale: Dsa indexer loss scale
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        selected_main_attn_dist,
        index_score,
        topk_indices,
        loss_scale,
    ):
        index_score = F.softmax(index_score, dim=-1, dtype=torch.float32)
        # considering only the selected token
        selected_main_attn_dist = F.normalize(selected_main_attn_dist, p=1, dim=-1)
        loss = (
            F.kl_div(
                (index_score + 1e-8).log(),
                selected_main_attn_dist + 1e-8,
                reduction="none",
            )
            .sum(dim=-1)
            .mean()
        )
        loss *= loss_scale
        return loss


def get_attn_scores(
    query,
    key,
    attention_mask,
    attn_scale,
):
    """aggregate the main attention scores"""
    num_head_q = query.shape[1]
    num_head_k = key.shape[1]
    if num_head_q != num_head_k and num_head_k != 1:
        raise NotImplementedError(
            f"Only support num_head_q == num_head_k or num_head_k == 1. "
            f"Current {num_head_q=}, {num_head_k=}."
        )

    attn = (query @ key.transpose(-1, -2)) * attn_scale

    if attention_mask is not None:
        attn.masked_fill_(attention_mask, float("-inf"))

    attn = F.softmax(attn, dim=-1, dtype=torch.float32)
    attn = attn.sum(dim=1)
    return attn


class DSV32_SDPA(torch.nn.Module):  # noqa: N801
    """Wrapper around `F.scaled_dot_product_attention` to make it CP compatible.

    This wrapper is needed because `F.scaled_dot_product_attention` is not
    a torch.nn.Module, and thus cannot be applied with _ContextParallel.
    We need to wrap it into a torch.nn.Module.

    Note:
        The forward function must have q, k, v as the first three arguments to be
        compatible with _ContextParallel.
    """

    sdpa_backends: ClassVar[list[SDPBackend]] = []

    def __init__(self, model_args: DeepSeekV32ModelArgs) -> None:
        super().__init__()
        self.model_args = model_args
        self.compute_dsa_indexer_loss = DSAIndexerLoss()
        if not self.sdpa_backends:
            # pyrefly: ignore [read-only]
            self.sdpa_backends = [
                SDPBackend.CUDNN_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.EFFICIENT_ATTENTION,
            ]

    def forward(
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = q.size(0)

        if q_indexer is None:
            with sdpa_kernel(self.sdpa_backends, set_priority=True):
                output = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    scale=scale,
                    is_causal=(attn_mask is None),
                )
            return q.new_zeros(1), output

        # prepare attention_mask for sparse attention according to lightning_indexer score
        index_score = bf16_index(
            # pyrefly: ignore [missing-attribute]
            q_indexer.contiguous(),
            # pyrefly: ignore [missing-attribute]
            weights.unsqueeze(-1),
            # pyrefly: ignore [missing-attribute]
            k_indexer.contiguous(),
        )

        seqlen = index_score.size(1)
        if q.size(2) != seqlen:
            q = q.narrow(2, 0, seqlen)
        if k.size(2) != seqlen:
            k = k.narrow(2, 0, seqlen)
        if v.size(2) != seqlen:
            v = v.narrow(2, 0, seqlen)
        if attn_mask is None:
            attn_mask = torch.where(
                torch.triu(
                    torch.ones((bsz, seqlen, seqlen), dtype=q.dtype, device=q.device),
                    diagonal=1,
                )
                == 1,
                float("-inf"),
                0.0,
            )
        index_score += attn_mask
        # pyrefly: ignore [bad-argument-type]
        topk_score, topk_indices = index_score.topk(min(index_topk, end_pos), dim=-1)
        query_positions = (
            torch.arange(seqlen, device=topk_indices.device).unsqueeze(0).unsqueeze(-1)
        )
        valid_positions = topk_indices <= query_positions
        # pyrefly: ignore [no-matching-overload]
        topk_indices = torch.where(
            valid_positions, topk_indices, torch.full_like(topk_indices, -1)
        )
        index_mask = torch.full(
            (bsz, seqlen, seqlen), float("-inf"), device=q.device
        ).scatter_(-1, topk_indices, 0)
        attention_masks = attn_mask + index_mask
        attention_masks = torch.isinf(attention_masks) & (attention_masks < 0)
        attention_masks = attention_masks.unsqueeze(1)

        # compute sparse attention
        with sdpa_kernel(self.sdpa_backends, set_priority=True):
            attn_mask = ~attention_masks
            output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, scale=scale, is_causal=False
            )

        # compute sparse lightning_indexer loss
        q_det = q.detach()
        k_det = k.detach()
        main_attn_dist = get_attn_scores(q_det, k_det, attention_masks, scale)
        selected_main_attn_dist = torch.gather(
            main_attn_dist, dim=-1, index=topk_indices
        )
        loss = self.compute_dsa_indexer_loss(
            selected_main_attn_dist,
            topk_score,
            topk_indices,
            1.0,
        )
        # pyrefly: ignore [bad-return]
        return loss, output


DSASparseAttention = DSV32_SDPA


class PreAttention(nn.Module):
    """
    Multi-head attention (MLA) module.
    """

    def __init__(self, model_args: DeepSeekV32ModelArgs):
        super().__init__()
        self.dim = model_args.dim
        self.n_heads = model_args.n_heads
        self.q_lora_rank = model_args.q_lora_rank
        self.kv_lora_rank = model_args.kv_lora_rank
        self.qk_nope_head_dim = model_args.qk_nope_head_dim
        self.qk_rope_head_dim = model_args.qk_rope_head_dim
        self.qk_head_dim = model_args.qk_nope_head_dim + model_args.qk_rope_head_dim
        self.v_head_dim = model_args.v_head_dim
        self.enable_mla_absorb = model_args.enable_mla_absorb
        self.n_layers = model_args.n_layers

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = nn.RMSNorm(self.q_lora_rank, eps=model_args.norm_eps)
        self.wq_b = nn.Linear(
            self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False
        )
        self.wkv_a = nn.Linear(
            self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False
        )
        self.kv_norm = nn.RMSNorm(self.kv_lora_rank, eps=model_args.norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.softmax_scale = self.qk_head_dim**-0.5

        if model_args.max_seq_len > model_args.original_seq_len:
            mscale = 0.1 * model_args.mscale * math.log(model_args.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        self.indexer = Indexer(model_args)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        layer_id,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Multi-Head Latent Attention (MLA) Layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        bsz, seqlen, _ = x.size()

        # Query projection
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.view(
            bsz, seqlen, -1, self.qk_head_dim
        )  # (bsz, seqlen, n_heads, qk_head_dim)
        q_nope, q_pe = torch.split(
            q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        q_pe = apply_rotary_emb(q_pe, freqs_cis, positions=positions)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # Key-value projection
        kv = self.wkv_a(x)  # (bsz, seqlen, kv_lora_rank + qk_rope_head_dim)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(
            k_pe.unsqueeze(2), freqs_cis, positions=positions
        )  # (bsz, seqlen, 1, qk_rope_head_dim)

        if not self.enable_mla_absorb:
            kv = self.wkv_b(
                self.kv_norm(kv)
            )  # (bsz, seqlen, n_heads * (qk_nope_head_dim + v_head_dim))
            kv = kv.view(bsz, seqlen, -1, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(
                kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1
            )
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_heads, -1)], dim=-1)

            q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
            k = k.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
            v = v.transpose(1, 2)  # (bsz, n_heads, seqlen, v_head_dim)

            q_indexer, weights, k_indexer, end_pos = self.indexer(
                x.detach(), qr.detach(), 0, freqs_cis, attention_masks, positions
            )
            return (
                q,
                k,
                v,
                attention_masks,
                self.softmax_scale,
                q_indexer,
                k_indexer,
                weights,
                end_pos,
                self.indexer.index_topk,
                None,
            )
        else:
            kv = self.kv_norm(kv)
            wkv_b_weight = self.wkv_b.weight.reshape(
                -1, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
            )
            w_uk = wkv_b_weight[:, : self.qk_nope_head_dim, :]
            w_uv = wkv_b_weight[:, self.qk_nope_head_dim :, :]
            w_uv_t = w_uv.permute(0, 2, 1).contiguous()

            q_nope = torch.einsum("bshq,hqr->bshr", q_nope, w_uk)
            k_nope = kv.unsqueeze(2)
            v = kv.unsqueeze(2)

            k = torch.cat([k_nope, k_pe], dim=-1)
            q = torch.cat([q_nope, q_pe], dim=-1)

            q = q.transpose(1, 2)  # (bsz, n_heads, seqlen, qk_head_dim)
            k = k.transpose(1, 2)  # (bsz, 1, seqlen, qk_head_dim)
            v = v.transpose(1, 2)  # (bsz, 1, seqlen, v_head_dim)

            q_indexer, weights, k_indexer, end_pos = self.indexer(
                x.detach(), qr.detach(), 0, freqs_cis, attention_masks, positions
            )

            return (
                q,
                k,
                v,
                attention_masks,
                self.softmax_scale,
                q_indexer,
                k_indexer,
                weights,
                end_pos,
                self.indexer.index_topk,
                w_uv_t,
            )

    def init_weights(self, init_std: float):
        linear_list = [
            self.wkv_a,
            self.wkv_b,
        ]
        linear_list.extend([self.wq_a, self.wq_b])
        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=0.02)
        self.indexer.init_weights(init_std)
        self.kv_norm.reset_parameters()
        if self.q_lora_rank > 0:
            self.q_norm.reset_parameters()


class PostAttention(nn.Module):
    def __init__(self, model_args: DeepSeekV32ModelArgs):
        super().__init__()
        self.enable_mla_absorb = model_args.enable_mla_absorb
        self.wo = nn.Linear(
            model_args.n_heads * model_args.v_head_dim, model_args.dim, bias=False
        )
        self.n_layers = model_args.n_layers

    def forward(
        self,
        x: torch.Tensor,
        output: torch.Tensor,
        w_uv_t: torch.Tensor,
        loss: torch.Tensor,
        layer_id: int,
    ):
        bsz, seqlen, _ = x.size()
        if self.enable_mla_absorb:
            output = torch.einsum("bhsr,hrv->bhsv", output, w_uv_t)

        # Reshape and project output
        output = output.transpose(
            1, 2
        ).contiguous()  # (bsz, seqlen, n_heads, v_head_dim)
        output = output.view(bsz, seqlen, -1)  # (bsz, seqlen, n_heads * v_head_dim)
        output = self.wo(output)
        DSAIndexerLossLoggingHelper.save_loss_to_tracker(loss, layer_id, self.n_layers)
        output = DSAIndexerLossAutoScaler.apply(output, loss)
        return output  # (bsz, seqlen, dim)

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.wo.weight, mean=0.0, std=init_std)


class Attention(nn.Module):
    def __init__(self, model_args: DeepSeekV32ModelArgs):
        super().__init__()
        self.pre_attention = PreAttention(model_args)
        self.inner_attention = DSASparseAttention(model_args)
        self.post_attention = PostAttention(model_args)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        layer_id,
        positions: torch.Tensor | None = None,
    ):
        (
            q,
            k,
            v,
            attention_masks,
            softmax_scale,
            q_indexer,
            k_indexer,
            weights,
            end_pos,
            index_topk,
            w_uv_t,
        ) = self.pre_attention(x, freqs_cis, attention_masks, layer_id, positions)
        loss, output = self.inner_attention(
            q,
            k,
            v,
            attn_mask=attention_masks,
            scale=softmax_scale,
            q_indexer=q_indexer,
            k_indexer=k_indexer,
            weights=weights,
            end_pos=end_pos,
            index_topk=index_topk,
        )
        final_output = self.post_attention(x, output, w_uv_t, loss, layer_id)
        return final_output

    def init_weights(self, init_std: float):
        self.pre_attention.init_weights(init_std)
        self.post_attention.init_weights(init_std)


class TransformerBlockV32(TransformerBlock):
    """
    Transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, model_args: DeepSeekV32ModelArgs):
        super().__init__(layer_id, model_args)
        # pyrefly: ignore [bad-assignment]
        self.attention = Attention(model_args)

    # pyrefly: ignore [bad-param-name-override]
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """
        if residual is None:
            x, residual = self.attention_norm(x), x
        else:
            x = x + residual
            residual = x
            x = self.attention_norm(x)
        x = self.attention(x, freqs_cis, attention_masks, self.layer_id, positions)
        x = x + residual
        residual = x
        x = self.ffn_norm(x)
        if self.moe_enabled:
            x = self.moe(x)
        else:
            x = self.feed_forward(x)
        return x, residual


class DeepSeekV32Model(DeepSeekV3Model):
    """
    DeepSeek-V3.2 Transformer model with attention and feed-forward layers.
    """

    def __init__(self, model_args: DeepSeekV32ModelArgs):
        super().__init__(model_args)
        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = TransformerBlockV32(layer_id, model_args)
        self.model_args = model_args
        # pyrefly: ignore [bad-assignment]
        self.norm = RMSNorm(model_args.dim)
        # When MTP is enabled, the forward pass receives sequences of length
        # max_seq_len + num_mtp_modules, so freqs_cis must cover that extra range.
        if model_args.num_mtp_modules > 0:
            import dataclasses

            extended_args = dataclasses.replace(
                model_args,
                max_seq_len=model_args.max_seq_len + model_args.num_mtp_modules,
            )
            self.register_buffer(
                "freqs_cis", precompute_freqs_cis(extended_args), persistent=False
            )

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ):
        """
        Forward pass for the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices if pipeline parallelism is not enabled.
                If pipeline parallelism is enabled, this will be the input token indices
                for the ranks on the first pipeline stage. This will be the activation of the
                previous pipeline stage if the current rank is not on the first stage.
            attention_masks: Optional masks for flex/varlen attention.
            positions: Position ids for CP/TP; set by Trainer when context_parallel is enabled
                (see torchtitan v0.2.2 prepare_context_parallel_input). Defaults to None.

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
        """

        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        residual = None
        for layer in self.layers.values():
            h, residual = layer(h, residual, self.freqs_cis, attention_masks, positions)
        if residual is not None:
            h = h + residual
        h = self.norm(h) if self.norm is not None else h
        output = self.output(h.float()) if self.output is not None else h
        return output
