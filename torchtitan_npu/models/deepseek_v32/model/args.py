# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from dataclasses import dataclass
from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs


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
    save_expert_format: Optional[str] = None
    hf_save_dir: Optional[str] = None
    save_patch_enabled: bool = False
    moe_impl: str = "standard"