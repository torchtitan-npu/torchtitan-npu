# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from typing import Any

import torch.distributed.checkpoint as dcp

from torchtitan.models.llama4 import Llama4StateDictAdapter

from torchtitan_npu.tools.weight_utils import convert_expert_format

MODEL = "model"


def dcp_load(
    self,
    state_dict: dict[str, Any],
    checkpoint_id: str,
    from_hf: bool,
    from_quantized: bool,
) -> None:
    """Load the checkpoint with dcp.
    Args:
        state_dict (dict): The state dict to load.
        checkpoint_id (str): The checkpoint id to load.
        from_hf (bool): Whether to load from HuggingFace checkpoint with
            its own model definition and safetensors format.

    We found that the Llama4 model encounters a bug during DCP checkpoint loading where certain weight keys are missing.
    To address this issue, we copied the relevant code from the original repository and made some modifications.
    """

    if from_hf:
        hf_state_dict = self.sd_adapter.to_hf(state_dict)
        hf_storage_reader = self.sd_adapter.get_hf_storage_reader(
            checkpoint_id, from_quantized
        )

        dcp.load(
            hf_state_dict,
            storage_reader=hf_storage_reader,
        )

        state_dict = self.sd_adapter.from_hf(hf_state_dict)
        self.states[MODEL].load_state_dict(state_dict)
    else:
        # To address the missing weight keys issue, perform a state dict conversion before calling dcp_load
        hf_state_dict = self.sd_adapter.to_hf(state_dict)
        state_dict = self.sd_adapter.from_hf(hf_state_dict)
        dcp.load(state_dict, checkpoint_id=checkpoint_id)

        if MODEL in self.states:
            self.states[MODEL].load_state_dict(state_dict)


class Llama4StateDictAdapterNpu(Llama4StateDictAdapter):
    def __init__(self, model_args, hf_assets_path: str | None = None):
        super().__init__(model_args, hf_assets_path)

        self._input_format = "hf"
        self.use_gmm = getattr(model_args.moe_args, "use_grouped_mm", False)

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
