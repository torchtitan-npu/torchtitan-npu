# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__all__ = [
    "parallelize_deepseekv32",
    "DeepSeekV32ModelArgs",
    "DeepSeekV32Model",
    "deepseekv32_args",
]

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers_with_moe_load_balancing
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.models.moe import MoEArgs
from torchtitan.protocols.train_spec import TrainSpec

from .infra.parallelize import parallelize_deepseekv32
from .model.args import DeepSeekV32ModelArgs
from .model.model import DeepSeekV32Model
from .model.state_dict_adapter import DeepSeekV32StateDictAdapter


deepseekv32_args = {
    "debugmodel": DeepSeekV32ModelArgs(
        vocab_size=129280,
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=1,
        n_dense_layers=1,
        n_heads=128,
        moe_args=MoEArgs(
            num_experts=8,
            num_shared_experts=1,
            top_k=8,
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
        # Checkpoint
        save_format="hf",
        save_expert_format="gmm",
        hf_save_dir=None,
        save_patch_enabled=False,
    ),
    "tinymodel": DeepSeekV32ModelArgs(
        vocab_size=129280,
        dim=256,
        inter_dim=512,
        moe_inter_dim=256,
        n_layers=1,
        n_dense_layers=1,
        n_heads=4,
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
        # Checkpoint
        save_format="hf",
        save_expert_format="gmm",
        hf_save_dir=None,
        save_patch_enabled=False,
    ),
    "671B_debug_4_layers": DeepSeekV32ModelArgs(
        vocab_size=129280,
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=4,
        n_dense_layers=3,
        n_heads=128,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="sigmoid",
            route_norm=True,
            route_scale=2.5,
            score_before_experts=False,
            use_grouped_mm=True,
        ),
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        attn_mask_type="block_causal",
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
    ),
    "671B_debug_128die": DeepSeekV32ModelArgs(
        vocab_size=129280,
        dim=7168,
        inter_dim=18432,
        moe_inter_dim=2048,
        n_layers=61,
        n_dense_layers=3,
        n_heads=128,
        moe_args=MoEArgs(
            num_experts=256,
            num_shared_experts=1,
            top_k=8,
            score_func="sigmoid",
            route_norm=True,
            route_scale=2.5,
            score_before_experts=False,
            use_grouped_mm=True,
        ),
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        index_n_heads=64,
        index_head_dim=128,
        index_topk=2048,
        save_format="hf",
        save_expert_format="gmm",
        hf_save_dir=None,
        save_patch_enabled=False,
    ),
}


def get_train_spec() -> TrainSpec:
    return TrainSpec(
        model_cls=DeepSeekV32Model,
        model_args=deepseekv32_args,
        parallelize_fn=parallelize_deepseekv32,
        pipelining_fn=pipeline_llm,
        build_optimizers_fn=build_optimizers_with_moe_load_balancing,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        state_dict_adapter=DeepSeekV32StateDictAdapter,
    )
