# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass

import torch
import torch_npu

from torchtitan.models.deepseek_v3.model.model import precompute_freqs_cis
from torchtitan.models.moe.moe import MoEArgs

from torchtitan_npu.models.deepseek_v32.model.args import DeepSeekV32ModelArgs
from torchtitan_npu.models.deepseek_v32.model.model import apply_rotary_emb, Attention


@dataclass
class AttentionTensorState:
    x: torch.Tensor
    freqs_cis: torch.Tensor
    qr: torch.Tensor
    q_nope: torch.Tensor
    q_pe: torch.Tensor
    kv: torch.Tensor
    k_pe: torch.Tensor


@dataclass
class DsaKernelInputs:
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    query_indexer: torch.Tensor
    weights: torch.Tensor
    key_indexer: torch.Tensor
    topk_indices: torch.Tensor
    query_rope: torch.Tensor
    key_rope: torch.Tensor


def _build_dsa_args(seq_len):
    args = DeepSeekV32ModelArgs(
        vocab_size=129280,
        dim=256,
        inter_dim=512,
        moe_inter_dim=256,
        n_layers=1,
        n_dense_layers=1,
        n_heads=64,
        moe_args=MoEArgs(
            num_experts=4,
            num_shared_experts=1,
            top_k=2,
            score_func="sigmoid",
            route_norm=True,
            score_before_experts=False,
            use_grouped_mm=False,
        ),
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
    )
    args.max_seq_len = max(args.max_seq_len, seq_len)
    return args


def _build_attention_tensors(attention, args, batch_size, seq_len, device):
    x = torch.zeros(batch_size, seq_len, args.dim, dtype=torch.bfloat16, device=device)
    freqs_cis = precompute_freqs_cis(args).to(device)[:seq_len]
    qr = attention.q_norm(attention.wq_a(x))
    q = attention.wq_b(qr).view(batch_size, seq_len, -1, attention.qk_head_dim)
    q_nope, q_pe = torch.split(
        q,
        [attention.qk_nope_head_dim, attention.qk_rope_head_dim],
        dim=-1,
    )
    q_pe = apply_rotary_emb(q_pe, freqs_cis)
    kv = attention.wkv_a(x)
    kv, k_pe = torch.split(
        kv, [attention.kv_lora_rank, attention.qk_rope_head_dim], dim=-1
    )
    k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
    return AttentionTensorState(
        x=x,
        freqs_cis=freqs_cis,
        qr=qr,
        q_nope=q_nope,
        q_pe=q_pe,
        kv=attention.kv_norm(kv),
        k_pe=k_pe,
    )


def _build_query_key_tensors(attention, attention_state):
    wkv_b_weight = attention.wkv_b.weight.reshape(
        -1, attention.qk_nope_head_dim + attention.v_head_dim, attention.kv_lora_rank
    )
    w_uk = wkv_b_weight[:, : attention.qk_nope_head_dim, :]
    query = torch.einsum("bshq,hqr->bshr", attention_state.q_nope, w_uk)
    key = attention_state.kv.unsqueeze(2)
    value = attention_state.kv.unsqueeze(2)
    if query.shape[2] not in (64, 128):
        raise AssertionError(
            f"DSA helper produced invalid query head count: {query.shape}"
        )
    return query, key, value


def _build_indexer_outputs(attention, attention_state):
    query_indexer, weights, key_indexer, _ = attention.indexer(
        attention_state.x.detach(),
        attention_state.qr.detach(),
        0,
        attention_state.freqs_cis,
        None,
    )
    topk_indices, _ = torch_npu.npu_lightning_indexer(
        query_indexer.detach(),
        key_indexer.detach(),
        weights.detach(),
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        layout_query="BSND",
        layout_key="BSND",
        sparse_count=attention.indexer.index_topk,
        sparse_mode=3,
        return_value=True,
    )
    return DsaKernelInputs(
        query=None,
        key=None,
        value=None,
        query_indexer=query_indexer,
        weights=weights,
        key_indexer=key_indexer,
        topk_indices=topk_indices.to(torch.int32),
        query_rope=None,
        key_rope=None,
    )


def _build_softmax_stats(attention, kernel_inputs, batch_size, seq_len):
    actual_seq_len = torch.full(
        (batch_size,),
        seq_len,
        dtype=torch.int32,
        device=kernel_inputs.query.device,
    )
    _, softmax_max, softmax_sum, *_ = torch_npu.npu_sparse_flash_attention(
        kernel_inputs.query.detach(),
        kernel_inputs.key.detach(),
        kernel_inputs.value.detach(),
        sparse_indices=kernel_inputs.topk_indices,
        block_table=None,
        actual_seq_lengths_query=actual_seq_len,
        actual_seq_lengths_kv=actual_seq_len,
        query_rope=kernel_inputs.query_rope.detach(),
        key_rope=kernel_inputs.key_rope.detach(),
        scale_value=attention.softmax_scale,
        sparse_block_size=1,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=True,
    )
    return softmax_max.detach(), softmax_sum.detach()


def run_lightning_indexer_smoke(npu_device, *, batch_size=1, seq_len=128):
    q_indexer = torch.randn(
        batch_size, seq_len, 64, 128, dtype=torch.bfloat16, device=npu_device
    )
    k_indexer = torch.randn(
        batch_size, seq_len, 1, 128, dtype=torch.bfloat16, device=npu_device
    )
    weights = torch.randn(
        batch_size, seq_len, 64, dtype=torch.bfloat16, device=npu_device
    )
    return torch_npu.npu_lightning_indexer(
        q_indexer,
        k_indexer,
        weights,
        actual_seq_lengths_query=None,
        actual_seq_lengths_key=None,
        layout_query="BSND",
        layout_key="BSND",
        sparse_count=16,
        sparse_mode=3,
        return_value=True,
    )


def build_model_backed_dsa_inputs(
    device, *, batch_size=1, seq_len=2048, requires_grad=False
):
    args = _build_dsa_args(seq_len)
    attention = Attention(args).to(device=device, dtype=torch.bfloat16)
    attention.eval()
    attention_state = _build_attention_tensors(
        attention, args, batch_size, seq_len, device
    )
    query, key, value = _build_query_key_tensors(attention, attention_state)
    kernel_inputs = _build_indexer_outputs(attention, attention_state)
    kernel_inputs.query = query
    kernel_inputs.key = key
    kernel_inputs.value = value
    kernel_inputs.query_rope = attention_state.q_pe
    kernel_inputs.key_rope = attention_state.k_pe
    softmax_max, softmax_sum = _build_softmax_stats(
        attention, kernel_inputs, batch_size, seq_len
    )

    query_indexer = kernel_inputs.query_indexer.detach()
    key_indexer = kernel_inputs.key_indexer.detach()
    weights = kernel_inputs.weights.detach()
    if requires_grad:
        query_indexer.requires_grad_()
        key_indexer.requires_grad_()
        weights.requires_grad_()

    return {
        "query": query.detach(),
        "key": key.detach(),
        "query_indexer": query_indexer,
        "key_indexer": key_indexer,
        "weights": weights,
        "topk_indices": kernel_inputs.topk_indices,
        "softmax_max": softmax_max,
        "softmax_sum": softmax_sum,
        "query_rope": attention_state.q_pe.detach(),
        "key_rope": attention_state.k_pe.detach(),
    }
