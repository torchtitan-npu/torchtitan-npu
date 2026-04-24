# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Optional, Tuple

from torch import nn
from torchtitan.config import JobConfig
from torchtitan.models.utils import get_moe_model_nparams_and_flops
from torchtitan.protocols.model import BaseModelArgs
from torchtitan.tools.logging import logger
from torchtitan.tools.utils import has_cuda_capability

from .moe import MoEArgs

# Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
@dataclass
class DeepSeekV4ModelArgs(BaseModelArgs):
    norm_eps: float = 1e-6
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 512
    enable_indexer_loss: bool = True
    save_format: str = "dcp"
    save_expert_format: Optional[str] = None
    hf_save_dir: Optional[str] = None
    save_patch_enabled: bool = False
    dim: int = 4096
    moe_args: MoEArgs = field(default_factory=MoEArgs)
    rope_head_dim: int = 64
    index_n_heads: int = 64
    q_lora_rank: int = 1024
    max_batch_size: int = 4
    max_seq_len: int = 4096
    n_heads: int = 64
    o_lora_rank: int = 1024
    head_dim: int = 512
    o_groups: int = 8
    window_size: int = 128
    compress_ratios: Tuple[int] = (
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
    )
    _debug_force_load_balance: bool = False
    hc_sinkhorn_iters: int = 20
    hc_mult: int = 4
    hc_eps: float = 1e-6
    vocab_size: int = 129280
    moe_inter_dim: int = 2048
    load_balance_coeff: float = 1e-3
    compress_rope_theta: float = 40000.0
    original_seq_len: int = 65536
    rope_theta: int = 10000
    rope_factor: int = 4
    beta_fast: int = 32
    beta_slow: int = 1
    n_layers: int = 4
    use_sfa: bool = False
    num_mtp_modules: int = 0
    mtp_layer_compress_ratio: int = 1

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        seq_len = job_config.training.seq_len
        if seq_len > self.max_seq_len:
            logger.warning(
                f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
            )
        self.max_seq_len = seq_len

        if (
            job_config.parallelism.context_parallel_degree > 1
            and self.attn_type != "sdpa"
        ):
            raise NotImplementedError("CP support is only supported for SDPA.")

        self.moe_args._debug_force_load_balance = (
            job_config.debug.moe_force_load_balance
        )

        # Configure expert parallel communication backend from config (defaults to "standard")
        self.moe_impl = job_config.parallelism.expert_parallel_comm_backend

        self.use_sfa = "deepseek_v4_sfa" in job_config.model.converters

        self.num_mtp_modules = job_config.training.num_mtp_modules

    def get_nparams_and_flops(self, model: nn.Module, seq_len: int) -> tuple[int, int]:
        return get_moe_model_nparams_and_flops(
            self,
            model,
            self.q_lora_rank + self.q_lora_rank,
            seq_len,
        )
