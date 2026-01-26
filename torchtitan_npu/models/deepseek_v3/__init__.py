# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved
# Copyright (c) Meta Platforms Inc. and affiliates
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import JobConfig
from torchtitan.models import deepseek_v3
from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
from torchtitan.models.moe import MoEArgs


deepseek_v3.deepseekv3_args["671B_debug"] = DeepSeekV3ModelArgs(
    vocab_size=129280,
    dim=7168,
    inter_dim=18432,
    moe_inter_dim=2048,
    n_layers=2,
    n_dense_layers=1,
    n_heads=128,
    moe_args=MoEArgs(
        num_experts=8,
        num_shared_experts=1,
        top_k=8,
        score_func="sigmoid",
        route_norm=True,
        route_scale=2.5,
        score_before_experts=False,
    ),
    n_expert_groups=8,
    n_limited_groups=1,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    use_flex_attn=False,
    attn_mask_type="block_causal",
)

deepseek_v3.deepseekv3_args["671B_debug_16die"] = DeepSeekV3ModelArgs(
    vocab_size=129280,
    dim=7168,
    inter_dim=18432,
    moe_inter_dim=2048,
    n_layers=61,
    n_dense_layers=3,
    n_heads=128,
    moe_args=MoEArgs(
        num_experts=32,
        num_shared_experts=1,
        top_k=8,
        score_func="sigmoid",
        route_norm=True,
        route_scale=2.5,
        score_before_experts=False,
    ),
    n_expert_groups=8,
    n_limited_groups=4,
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    use_flex_attn=False,
    attn_mask_type="block_causal",
)


def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
    seq_len = job_config.training.seq_len
    if seq_len > self.max_seq_len:
        logger.warning(
            f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
        )
    self.max_seq_len = seq_len

    if job_config.parallelism.context_parallel_degree > 1 and self.use_flex_attn:
        raise NotImplementedError("CP support for FlexAttention is still in progress.")

    self.moe_args._debug_force_load_balance = (
        job_config.training.debug_moe_force_load_balance
    )


DeepSeekV3ModelArgs.update_from_config = update_from_config
