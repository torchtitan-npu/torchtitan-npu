# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any, Dict, Optional

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor
from torchtitan.models.deepseek_v3 import DeepSeekV3StateDictAdapter

from torchtitan_npu.tools.weight_utils import (
    convert_expert_format,
    detect_input_format_by_path,
)


logger = logging.getLogger(__name__)


class DeepSeekV4StateDictAdapter(DeepSeekV3StateDictAdapter):
    def __init__(self, model_args, hf_assets_path: Optional[str] = None):
        super().__init__(model_args, hf_assets_path)

        # key mapping
        self.from_hf_map = {
            "embed.weight": "tok_embeddings.weight",
            # Attention Module
            "layers.{}.attn.attn_sink": "layers.{}.attention.inner_attention.attn_sink",
            "layers.{}.attn.kv_norm.weight": "layers.{}.attention.pre_attention.kv_norm.weight",
            "layers.{}.attn.q_norm.weight": "layers.{}.attention.pre_attention.q_norm.weight",
            "layers.{}.attn.wo_a.weight": "layers.{}.attention.post_attention.wo_a.weight",
            "layers.{}.attn.wkv.weight": "layers.{}.attention.pre_attention.wkv.weight",
            "layers.{}.attn.wo_b.weight": "layers.{}.attention.post_attention.wo_b.weight",
            "layers.{}.attn.wq_a.weight": "layers.{}.attention.pre_attention.wq_a.weight",
            "layers.{}.attn.wq_b.weight": "layers.{}.attention.pre_attention.wq_b.weight",
            # Transformer Layer
            "layers.{}.attn_norm.weight": "layers.{}.attention_norm.weight",
            "layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
            # MoE Module
            "layers.{}.ffn.experts.{}.w1.weight": "layers.{}.moe.experts.w1",
            "layers.{}.ffn.experts.{}.w3.weight": "layers.{}.moe.experts.w3",
            "layers.{}.ffn.experts.{}.w2.weight": "layers.{}.moe.experts.w2",
            "layers.{}.ffn.gate.weight": "layers.{}.moe.router.gate.weight",
            "layers.{}.ffn.gate.bias": "layers.{}.moe.expert_bias",
            "layers.{}.ffn.shared_experts.w1.weight": "layers.{}.moe.shared_experts.w1.weight",
            "layers.{}.ffn.shared_experts.w3.weight": "layers.{}.moe.shared_experts.w3.weight",
            "layers.{}.ffn.shared_experts.w2.weight": "layers.{}.moe.shared_experts.w2.weight",
            # mHC
            "layers.{}.hc_attn_base": "layers.{}.hc_attn_base",
            "layers.{}.hc_attn_fn": "layers.{}.hc_attn_fn",
            "layers.{}.hc_attn_scale": "layers.{}.hc_attn_scale",
            "layers.{}.hc_ffn_base": "layers.{}.hc_ffn_base",
            "layers.{}.hc_ffn_fn": "layers.{}.hc_ffn_fn",
            "layers.{}.hc_ffn_scale": "layers.{}.hc_ffn_scale",
            "hc_head_base": "hc_head_base",
            "hc_head_fn": "hc_head_fn",
            "hc_head_scale": "hc_head_scale",
            "norm.weight": "norm.weight",
            "head.weight": "output.weight",
        }
        self.compress_ratios = model_args.compress_ratios
        for layer_id in range(model_args.n_layers):
            cr = self.compress_ratios[layer_id]
            # compressor
            if cr != 1:
                compressor_attr = "compressor" if cr == 4 else "compressor_128"
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.attn.compressor.ape": f"layers.{layer_id}.attention.pre_attention.{compressor_attr}.ape",
                        f"layers.{layer_id}.attn.compressor.norm.weight": f"layers.{layer_id}.attention.pre_attention.{compressor_attr}.norm.weight",
                        f"layers.{layer_id}.attn.compressor.wgate.weight": f"layers.{layer_id}.attention.pre_attention.{compressor_attr}.wgate.weight",
                        f"layers.{layer_id}.attn.compressor.wkv.weight": f"layers.{layer_id}.attention.pre_attention.{compressor_attr}.wkv.weight",
                    }
                )
            # indexer
            if cr == 4:
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.attn.indexer.compressor.ape": f"layers.{layer_id}.attention.pre_attention.indexer.compressor.ape",
                        f"layers.{layer_id}.attn.indexer.compressor.norm.weight": f"layers.{layer_id}.attention.pre_attention.indexer.compressor.norm.weight",
                        f"layers.{layer_id}.attn.indexer.compressor.wgate.weight": f"layers.{layer_id}.attention.pre_attention.indexer.compressor.wgate.weight",
                        f"layers.{layer_id}.attn.indexer.compressor.wkv.weight": f"layers.{layer_id}.attention.pre_attention.indexer.compressor.wkv.weight",
                        f"layers.{layer_id}.attn.indexer.wq_b.weight": f"layers.{layer_id}.attention.pre_attention.indexer.wq_b.weight",
                        f"layers.{layer_id}.attn.indexer.weights_proj.weight": f"layers.{layer_id}.attention.pre_attention.indexer.weights_proj.weight",
                    }
                )
            if layer_id <= model_args.moe_args.n_hash_layers - 1:
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.ffn.gate.tid2eid": f"layers.{layer_id}.moe.router.tid2eid",
                    }
                )
        if model_args.num_mtp_modules > 0:
            for mtp_layer_id in range(model_args.num_mtp_modules):
                layer_id = mtp_layer_id + model_args.n_layers
                self.from_hf_map.update(
                    {
                        "layers.{}.enorm.weight": "layers.{}.enorm.weight",
                        "layers.{}.hnorm.weight": "layers.{}.hnorm.weight",
                        "layers.{}.e_proj.weight": "layers.{}.e_proj.weight",
                        "layers.{}.h_proj.weight": "layers.{}.h_proj.weight",
                    }
                )
        # configs
        self.use_gmm = getattr(model_args.moe_args, "use_grouped_mm", False)
        self.n_experts = model_args.moe_args.num_experts
        self.first_k_dense = getattr(model_args.moe_args, "first_k_dense", 0)
        self._input_format = "hf"
        self._input_expert_format = "standard"
        self._to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        self._to_hf_passthrough = (
            "tid2eid",
            "compressor",
            "indexer",
        )
        self._from_hf_passthrough = (
            "tid2eid",
            "compressor",
            "indexer",
        )

        # apply checkpoint patch
        self._setup_checkpoint_patch(model_args)

    def to_hf_mtp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        new_state_dict = {}
        for key, tensor in state_dict.items():
            match = re.match(r"layers\.(\d+)\.(.+)", key)
            if match:
                num = int(match.group(1))
                rest = match.group(2)

                if num >= self.model_args.n_layers:
                    new_key = f"mtp.0.{rest}"
                else:
                    new_key = key
            else:
                new_key = key
            new_state_dict[new_key] = tensor
        return new_state_dict

    def to_hf_new(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split the GroupedExperts' weight into separate expert's weight.
        """
        to_hf_map = self._to_hf_map
        passthrough = self._to_hf_passthrough
        hf_state_dict = {}
        for key, value in state_dict.items():
            if any(token in key for token in passthrough):
                new_key = to_hf_map[key]
                # uses torch.finfo() which only supports float types. Convert to float
                # here and convert back in from_hf_new on load.
                if "tid2eid" in key:
                    value = value.to(torch.float32)
                hf_state_dict[new_key] = value
            elif "moe.experts" in key:
                abstract_key = self._get_abstract_key(key, count=1)
                layer_num = self._first_number(key)
                new_abstract_key = to_hf_map[abstract_key]

                # Store the GroupedExperts Weight metadata for from_hf()
                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    # Split GroupedExperts weight to local individual expert weights
                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)

                else:
                    # keep this path for offline conversion
                    split_values = self._split_experts_weights(
                        value, self.model_args.moe_args.num_experts
                    )

                    for expert_num in range(0, self.model_args.moe_args.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                hf_state_dict[self._map_layer_key(key, to_hf_map)] = value

            else:
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        hf_state_dict = self.to_hf_mtp(hf_state_dict)

        return hf_state_dict

    def get_hf_storage_reader(self, path: str, from_quantized: bool = False):
        self._input_format = detect_input_format_by_path(path)

        if self._input_format == "hf":
            return HuggingFaceStorageReader(path)
        else:
            from torch.distributed.checkpoint import FileSystemReader

            return FileSystemReader(path)

    def to_hf(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Create a load plan/ Convert to HF format"""
        if self._input_format == "dcp":
            return state_dict

        has_w13 = any(".moe.experts.w13" in k for k in state_dict.keys())
        if has_w13:
            # split w13 -> w1, w3 for load plan
            working_state = self._split_w13_for_mapping(state_dict)
            return self.to_hf_new(working_state)
        else:
            return self.to_hf_new(state_dict)

    def from_hf(self, hf_state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert loaded data to runtime format"""
        filtered = {
            k: v
            for k, v in hf_state_dict.items()
            if not k.endswith(".weight_scale_inv")
        }

        if self._input_format == "hf":
            state_dict = self.from_hf_new(filtered)
        else:
            state_dict = filtered
        target = "gmm" if self.use_gmm else "standard"
        state_dict = convert_expert_format(state_dict, target)

        return state_dict

    def from_hf_mtp(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        new_state_dict = {}
        for key, tensor in state_dict.items():
            match = re.match(r"mtp\.(\d+)\.(.+)", key)
            if match:
                num = int(match.group(1))
                rest = match.group(2)
                new_key = f"layers.{self.model_args.n_layers}.{rest}"
            else:
                new_key = key
            new_state_dict[new_key] = tensor
        return new_state_dict

    def from_hf_new(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. When loading from HF checkpoint, dequantize the weights from float8 to float32.
        2. Convert between the HF shape and the torchtitan shape.
        3. Concat separate expert's weight into GroupedExperts' weight.
        """

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}
        passthrough = self._from_hf_passthrough

        hf_state_dict = self.from_hf_mtp(hf_state_dict)

        for key, value in hf_state_dict.items():
            if any(token in key for token in passthrough):
                new_key = self.from_hf_map[key]
                # tid2eid was saved as float32 to work around torch.finfo() issue
                # in safetensors consolidation. Convert back to int64.
                if "tid2eid" in key:
                    value = value.to(torch.int64)
                state_dict[new_key] = value
            elif "ffn.experts" in key:
                abstract_key = self._get_abstract_key(key, count=2)
                layer_num, expert_num, _ = re.findall(r"\d+", key)
                titan_abstract_key = self.from_hf_map[abstract_key]
                new_key = titan_abstract_key.format(layer_num)

                # Store the expert's weight in expert_weights_by_layer for concatenating later.
                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                # Use stored metadata to decide path (online vs offline)
                # Online mode: local_experts_indices was populated during to_hf()
                if titan_abstract_key in self.local_experts_indices:
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:  # keep this path to be compatible with offline conversion
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        self.model_args.moe_args.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                state_dict[self._map_layer_key(key, self.from_hf_map)] = value

            else:
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value

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

    def _split_w13_for_mapping(self, state_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Split w13 into w1 and w3 for HF mapping"""
        result = {}

        for key, value in state_dict.items():
            if ".moe.experts.w13" in key:
                base_key = key.replace(".w13", "")

                # Create placeholders w1 and w3
                # For DTensor, the shape needs to be adjusted.
                if isinstance(value, DTensor):
                    self._w13_placements = value.placements
                    self._w13_device_mesh = value.device_mesh

                    shape = value.shape
                    new_shape = (shape[0], shape[1] // 2, shape[2])

                    from torch.distributed.tensor import zeros as dt_zeros

                    w1 = dt_zeros(
                        new_shape,
                        device_mesh=value.device_mesh,
                        placements=value.placements,
                    )
                    w3 = dt_zeros(
                        new_shape,
                        device_mesh=value.device_mesh,
                        placements=value.placements,
                    )
                else:
                    half = value.shape[1] // 2
                    w1 = torch.empty(
                        value.shape[0],
                        half,
                        value.shape[2],
                        dtype=value.dtype,
                        device=value.device,
                    )
                    w3 = torch.empty(
                        value.shape[0],
                        half,
                        value.shape[2],
                        dtype=value.dtype,
                        device=value.device,
                    )

                result[base_key + ".w1"] = w1
                result[base_key + ".w3"] = w3
            else:
                result[key] = value

        return result

    @staticmethod
    def _get_abstract_key(key: str, count: int) -> str:
        return re.sub(r"(\d+)", "{}", key, count=count)

    @staticmethod
    def _first_number(key: str) -> str:
        return re.search(r"\d+", key).group(0)

    @classmethod
    def _map_layer_key(cls, key: str, mapping: dict[str, str]) -> str:
        abstract_key = cls._get_abstract_key(key, count=1)
        layer_num = cls._first_number(key)
        return mapping[abstract_key].format(layer_num)
