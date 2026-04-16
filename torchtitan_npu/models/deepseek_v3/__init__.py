# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torchtitan.config import JobConfig
from torchtitan.models import deepseek_v3
from torchtitan.models.deepseek_v3 import get_train_spec
from torchtitan.models.moe import MoEArgs

from torchtitan_npu.models.deepseek_v3.model.args import DeepSeekV3ModelArgs
from .infra.parallelize import _parallelize_deepseekv3_wrapper


logger = logging.getLogger(__name__)

deepseek_v3.deepseekv3_args["671B_debug"] = DeepSeekV3ModelArgs(
    vocab_size=129280,
    dim=128,
    inter_dim=512,
    moe_inter_dim=128,
    n_layers=4,
    n_dense_layers=3,
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
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
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
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
)

deepseek_v3.deepseekv3_args["671B_debug_128die"] = DeepSeekV3ModelArgs(
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
    ),
    q_lora_rank=1536,
    kv_lora_rank=512,
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
)


def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
    seq_len = job_config.training.seq_len
    if seq_len > self.max_seq_len:
        logger.warning(
            f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
        )
    self.max_seq_len = seq_len

    self.moe_impl = job_config.parallelism.expert_parallel_comm_backend
    self.moe_args._debug_force_load_balance = job_config.debug.moe_force_load_balance


def get_train_spec_wrapper():
    train_spec = get_train_spec()
    train_spec.parallelize_fn = _parallelize_deepseekv3_wrapper
    return train_spec


deepseek_v3.get_train_spec = get_train_spec_wrapper

DeepSeekV3ModelArgs.update_from_config = update_from_config
