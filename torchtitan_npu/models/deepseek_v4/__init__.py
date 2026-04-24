# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "parallelize_deepseek_v4",
    "DeepSeekV4ModelArgs",
    "DeepSeekV4Model",
    "deepseekv4_args",
]

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_deepseek_v4
from .model.args import DeepSeekV4ModelArgs
from .model.model import DeepSeekV4Model
from .model.moe import MoEArgs
from .model.state_dict_adapter import DeepSeekV4StateDictAdapter


deepseekv4_args = {
    "285B_debug_4_layers": DeepSeekV4ModelArgs(
        vocab_size=129280,
        n_layers=4,
        n_heads=64,
        max_batch_size=4,
        max_seq_len=4096,
        dim=4096,
        moe_inter_dim=2048,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        window_size=128,
        compress_ratios=(1, 1, 4, 128),
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=6,
            score_func="sqrtsoftplus",
            route_norm=True,
            score_before_experts=False,
            use_grouped_mm=True,
            n_hash_layers=3,
            swiglu_limit=10,
        ),
        hc_sinkhorn_iters=20,
        hc_mult=4,
        hc_eps=1e-6,
        compress_rope_theta=160000,
        original_seq_len=65536,
        rope_theta=10000,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        # Indexer
        enable_indexer_loss=True,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        # Checkpoint
        save_format="hf",
        save_expert_format="gmm",
        hf_save_dir=None,
    ),
    "285B_debug_43_layers": DeepSeekV4ModelArgs(
        vocab_size=129280,
        n_layers=43,
        n_heads=64,
        max_batch_size=4,
        max_seq_len=4096,
        dim=4096,
        moe_inter_dim=2048,
        head_dim=512,
        rope_head_dim=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        window_size=128,
        compress_ratios=(
            1,
            1,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
            128,
            4,
        ),
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=6,
            score_func="sqrtsoftplus",
            route_norm=True,
            score_before_experts=False,
            use_grouped_mm=True,
            n_hash_layers=3,
            swiglu_limit=10,
        ),
        hc_sinkhorn_iters=20,
        hc_mult=4,
        hc_eps=1e-6,
        compress_rope_theta=160000,
        original_seq_len=65536,
        rope_theta=10000,
        rope_factor=16,
        beta_fast=32,
        beta_slow=1,
        # Indexer
        enable_indexer_loss=True,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=512,
        # Checkpoint
        save_format="hf",
        save_expert_format="gmm",
        hf_save_dir=None,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=DeepSeekV4Model,
        model_args=deepseekv4_args,
        parallelize_fn=parallelize_deepseek_v4,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        state_dict_adapter=DeepSeekV4StateDictAdapter,
    )
