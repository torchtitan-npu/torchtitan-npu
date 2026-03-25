# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import torch

import torchtitan_npu.tools.weight_utils as weight_utils
from torchtitan_npu.tools.weight_utils import (
    convert_expert_format,
    detect_expert_format,
)


def test_detect_expert_format_returns_none_for_non_moe_weights():
    state_dict = {"layer.weight": torch.randn(4, 4)}

    assert detect_expert_format(state_dict) == "none"


def test_detect_expert_format_recognizes_standard_experts():
    state_dict = {"model.layers.0.moe.experts.w1": torch.randn(2, 4, 8)}

    assert detect_expert_format(state_dict) == "standard"


def test_detect_expert_format_recognizes_gmm_experts():
    state_dict = {"model.layers.0.moe.experts.w13": torch.randn(2, 8, 8)}

    assert detect_expert_format(state_dict) == "gmm"


def test_convert_expert_format_fuses_standard_weights_into_w13():
    w1 = torch.randn(2, 4, 8)
    w3 = torch.randn(2, 4, 8)
    state_dict = {
        "model.layers.0.moe.experts.w1": w1.clone(),
        "model.layers.0.moe.experts.w3": w3.clone(),
    }

    result = convert_expert_format(state_dict, "gmm")

    assert "model.layers.0.moe.experts.w13" in result
    assert result["model.layers.0.moe.experts.w13"].shape == (2, 8, 8)


def test_convert_expert_format_splits_w13_back_to_standard():
    w13 = torch.randn(2, 8, 8)
    state_dict = {"model.layers.0.moe.experts.w13": w13.clone()}

    result = convert_expert_format(state_dict, "standard")

    assert "model.layers.0.moe.experts.w1" in result
    assert "model.layers.0.moe.experts.w3" in result
    assert result["model.layers.0.moe.experts.w1"].shape == (2, 4, 8)
    assert result["model.layers.0.moe.experts.w3"].shape == (2, 4, 8)


def test_convert_expert_format_splits_dtensor_w13_with_placements(monkeypatch):
    from types import SimpleNamespace

    captured_calls = []

    class FakeDTensor:
        def __init__(self, local_tensor, *, device_mesh, placements):
            self._local_tensor = local_tensor
            self.device_mesh = device_mesh
            self.placements = placements

        @classmethod
        def from_local(cls, local_tensor, *, device_mesh, placements):
            captured_calls.append(
                {
                    "local_shape": tuple(local_tensor.shape),
                    "device_mesh": device_mesh,
                    "placements": placements,
                }
            )
            return cls(local_tensor, device_mesh=device_mesh, placements=placements)

        def to_local(self):
            return self._local_tensor

    monkeypatch.setattr(weight_utils, "DTensor", FakeDTensor)

    w13 = FakeDTensor(
        torch.randn(2, 8, 8),
        device_mesh=SimpleNamespace(name="mesh"),
        placements=("shard",),
    )
    state_dict = {"model.layers.0.moe.experts.w13": w13}

    result = convert_expert_format(state_dict, "standard")

    assert len(captured_calls) == 2
    assert captured_calls[0]["placements"] == ("shard",)
    assert captured_calls[1]["placements"] == ("shard",)
    assert result["model.layers.0.moe.experts.w1"].to_local().shape == (2, 4, 8)
    assert result["model.layers.0.moe.experts.w3"].to_local().shape == (2, 4, 8)
