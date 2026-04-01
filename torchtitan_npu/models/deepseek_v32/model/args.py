# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.config import JobConfig

from torchtitan_npu.models.deepseek_v3.model.args import DeepSeekV3ModelArgs


# Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
@dataclass
class DeepSeekV32ModelArgs(DeepSeekV3ModelArgs):

    norm_eps: float = 1e-6
    index_n_heads: int = 64
    index_head_dim: int = 128
    index_topk: int = 2048

    enable_mla_absorb: bool = True
    enable_indexer_loss: bool = True
    save_format: str = "dcp"
    save_expert_format: str | None = None
    hf_save_dir: str | None = None
    save_patch_enabled: bool = False
    # pyrefly: ignore [bad-override]
    moe_impl: str = "standard"
    num_mtp_modules: int = 0

    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        super().update_from_config(job_config)
        if hasattr(job_config.training, "num_mtp_modules"):
            self.num_mtp_modules = job_config.training.num_mtp_modules
