# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import logging
from typing import Any

from torchtitan.models.deepseek_v3 import DeepSeekV3StateDictAdapter

from torchtitan_npu.tools.weight_utils import (
    _split_w13_for_mapping,
    convert_expert_format,
)

logger = logging.getLogger(__name__)


class DeepSeekV3StateDictAdapterNpu(DeepSeekV3StateDictAdapter):
    def __init__(self, model_args, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)

        self.use_gmm = getattr(model_args.moe_args, "use_grouped_mm", False)
        self._input_format = "hf"
        self._input_expert_format = "standard"

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if self._input_format == "dcp":
            return state_dict
        has_w13 = any(".moe.experts.w13" in k for k in state_dict.keys())
        if has_w13:
            working_state = _split_w13_for_mapping(state_dict)
            return super().to_hf(working_state)
        else:
            return super().to_hf(state_dict)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert loaded data to runtime format"""
        filtered = {
            k: v
            for k, v in hf_state_dict.items()
            if not k.endswith(".weight_scale_inv")
        }

        if self._input_format == "hf":
            state_dict = super().from_hf(filtered)
        else:
            state_dict = filtered
        target = "gmm" if self.use_gmm else "standard"
        state_dict = convert_expert_format(state_dict, target)

        return state_dict
