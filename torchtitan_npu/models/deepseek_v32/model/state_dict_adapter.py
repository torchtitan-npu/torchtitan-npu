# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any

from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader

from torchtitan.models.deepseek_v3 import DeepSeekV3StateDictAdapter

from torchtitan_npu.tools.weight_utils import (
    _split_w13_for_mapping,
    convert_expert_format,
    detect_input_format_by_path,
)

logger = logging.getLogger(__name__)


class DeepSeekV32StateDictAdapter(DeepSeekV3StateDictAdapter):
    def __init__(self, model_args, hf_assets_path: str | None = None):
        super().__init__(model_args, hf_assets_path)

        # key mapping
        self._setup_v32_mappings(model_args)

        # configs
        self.use_gmm = getattr(model_args.moe_args, "use_grouped_mm", False)
        self.n_experts = model_args.moe_args.num_experts
        self.first_k_dense = getattr(model_args.moe_args, "first_k_dense", 0)
        self._input_format = "hf"
        self._input_expert_format = "standard"

        # apply checkpoint patch
        self._setup_checkpoint_patch(model_args)

    # pyrefly: ignore [bad-override]
    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        self._input_format = detect_input_format_by_path(path)

        if self._input_format == "hf":
            return HuggingFaceStorageReader(path)
        else:
            from torch.distributed.checkpoint import FileSystemReader

            return FileSystemReader(path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Create a load plan/ Convert to HF format"""
        if self._input_format == "dcp":
            return state_dict

        has_w13 = any(".moe.experts.w13" in k for k in state_dict.keys())
        if has_w13:
            # split w13 -> w1, w3 for load plan
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

    def _setup_checkpoint_patch(self, model_args):
        """setup checkpoint save patch"""
        try:
            from ....tools import checkpoint_patch

            checkpoint_patch.configure_from_model_args(model_args, adapter=self)

            if checkpoint_patch.is_enabled():
                success = checkpoint_patch.apply_patch()
                if success:
                    logger.info(
                        "Checkpoint save patch initialized from StateDict Adaptor"
                    )

        except Exception as e:
            logger.error(
                f"Failed to setup checkpoint patch, training will continue with original saving configs: {e}"
            )

    def _setup_v32_mappings(self, model_args):
        """Deepseek V32 key mapping"""
        # MLA:
        self.from_hf_map.pop("model.layers.{}.self_attn.q_proj.weight", None)
        self.from_hf_map.update(
            {
                "model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.wq_a.weight",
                "model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attention.q_norm.weight",
                "model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.wq_b.weight",
            }
        )

        # Indexer
        self.from_hf_map.update(
            {
                "model.layers.{}.self_attn.indexer.wq_b.weight": "layers.{}.attention.indexer.wq_b.weight",
                "model.layers.{}.self_attn.indexer.wk.weight": "layers.{}.attention.indexer.wk.weight",
                "model.layers.{}.self_attn.indexer.k_norm.weight": "layers.{}.attention.indexer.k_norm.weight",
                "model.layers.{}.self_attn.indexer.k_norm.bias": "layers.{}.attention.indexer.k_norm.bias",
                "model.layers.{}.self_attn.indexer.weights_proj.weight": "layers.{}.attention.indexer.weights_proj.weight",
            }
        )

        # MTP
        if model_args.num_mtp_modules > 0:
            self.from_hf_map.update(
                {
                    "model.layers.{}.enorm.weight": "layers.{}.enorm.weight",
                    "model.layers.{}.hnorm.weight": "layers.{}.hnorm.weight",
                    "model.layers.{}.eh_proj.weight": "layers.{}.eh_proj.weight",
                }
            )
