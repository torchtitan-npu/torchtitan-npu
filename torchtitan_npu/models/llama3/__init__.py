# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models import llama3
from torchtitan.models.llama3.model.args import TransformerModelArgs

llama3.llama3_args["8b_debug_16die"] = TransformerModelArgs(
    dim=4096,
    n_layers=16,
    n_heads=32,
    n_kv_heads=8,
    ffn_dim_multiplier=1.3,
    multiple_of=1024,
    rope_theta=500000,
)
