# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import sys
import types
from unittest.mock import patch

import torch

from torchtitan_npu.tools import checkpoint_patch


class _FakeCheckpointManager:
    last_save_call = None

    def __init__(self):
        self.enable_checkpoint = True
        self.interval = 10

    # Keep these as instance methods because checkpoint_patch wraps and calls
    # the captured originals with an explicit `self` argument.
    def save(self, curr_step, last_step=False):
        self.__class__.last_save_call = (curr_step, last_step)
        return "original"

    def _flattened_model_states_sd(self):
        assert self.enable_checkpoint is True
        return {"weight": torch.ones(2, 2)}


def _build_checkpoint_module():
    return types.SimpleNamespace(CheckpointManager=_FakeCheckpointManager)


def _patch_runtime(config):
    return patch.multiple(
        checkpoint_patch,
        _config=config,
        _original_save=None,
        _original_model_states_sd=None,
    )


def _apply_patch_with_fake_module(config):
    fake_module = _build_checkpoint_module()

    runtime_patch = _patch_runtime(config)
    module_patch = patch.dict(
        sys.modules,
        {"torchtitan.components.checkpoint": fake_module},
        clear=False,
    )

    return fake_module, runtime_patch, module_patch


def test_save_config_defaults_and_reset():
    config = checkpoint_patch.SaveConfig(
        enabled=True,
        save_format="hf",
        save_expert_format="standard",
        hf_save_dir="/tmp/hf",
        num_experts=8,
    )
    config.set_adapter(object())
    config.set_patched(True)

    config.reset()

    assert config.enabled is False
    assert config.save_format == "dcp"
    assert config.save_expert_format is None
    assert config.hf_save_dir is None
    assert config.num_experts == 0
    assert config.get_adapter() is None
    assert config.is_patched() is False


def test_configure_from_model_args_updates_runtime_config():
    config = checkpoint_patch.SaveConfig()
    model_args = types.SimpleNamespace(
        save_patch_enabled=True,
        save_format="hf",
        save_expert_format="standard",
        hf_save_dir="outputs/hf",
    )
    adapter = object()

    with patch.object(checkpoint_patch, "_config", config):
        checkpoint_patch.configure_from_model_args(model_args, adapter=adapter)
        assert checkpoint_patch.is_enabled() is True

    assert config.save_format == "hf"
    assert config.save_expert_format == "standard"
    assert config.hf_save_dir == "outputs/hf"
    assert config.get_adapter() is adapter


def test_apply_patch_wraps_model_state_export_when_expert_conversion_enabled(monkeypatch):
    config = checkpoint_patch.SaveConfig(
        enabled=True,
        save_format="dcp",
        save_expert_format="grouped",
    )
    converted = {"converted": torch.ones(1)}
    fake_module, runtime_patch, module_patch = _apply_patch_with_fake_module(config)

    monkeypatch.setattr(checkpoint_patch, "detect_expert_format", lambda sd: "standard")
    monkeypatch.setattr(
        checkpoint_patch,
        "convert_expert_format",
        lambda sd, target: converted,
    )

    with runtime_patch, module_patch:
        assert checkpoint_patch.apply_patch() is True
        manager = fake_module.CheckpointManager()
        result = getattr(manager, "_flattened_model_states_sd")()

    assert result is converted
    assert config.is_patched() is True


def test_apply_patch_wraps_save_and_preserves_original_dcp_behavior():
    config = checkpoint_patch.SaveConfig(enabled=True, save_format="dcp")
    fake_module, runtime_patch, module_patch = _apply_patch_with_fake_module(config)
    _FakeCheckpointManager.last_save_call = None

    with runtime_patch, module_patch:
        assert checkpoint_patch.apply_patch() is True
        manager = fake_module.CheckpointManager()
        result = manager.save(10, False)

    assert result == "original"
    assert _FakeCheckpointManager.last_save_call == (10, False)


def test_apply_patch_skips_save_when_step_does_not_match_interval():
    config = checkpoint_patch.SaveConfig(enabled=True, save_format="dcp")
    fake_module, runtime_patch, module_patch = _apply_patch_with_fake_module(config)
    _FakeCheckpointManager.last_save_call = None

    with runtime_patch, module_patch:
        assert checkpoint_patch.apply_patch() is True
        manager = fake_module.CheckpointManager()
        result = manager.save(3, False)

    assert result is None
    assert _FakeCheckpointManager.last_save_call is None


def test_apply_patch_returns_false_when_disabled():
    config = checkpoint_patch.SaveConfig()
    config.enabled = False

    with _patch_runtime(config):
        assert checkpoint_patch.apply_patch() is False


def test_convert_to_hf_and_save_excludes_expert_tensors_from_non_expert_branch(monkeypatch, tmp_path):
    class CountingTensor:
        def __init__(self, value):
            self.value = value
            self.full_tensor_calls = 0

        def full_tensor(self):
            self.full_tensor_calls += 1
            return self.value

    class Adapter:
        model_args = types.SimpleNamespace(moe_args=types.SimpleNamespace(num_experts=1))

        @staticmethod
        def to_hf(state_dict):
            return state_dict

    saved = {}
    expert_tensor = CountingTensor(torch.ones(2, 2))

    fake_safetensors = types.SimpleNamespace(
        save_file=lambda tensors, path: saved.update({"tensors": tensors, "path": path})
    )

    monkeypatch.setattr(checkpoint_patch.torch.distributed, "is_initialized", lambda: False)
    config = checkpoint_patch.SaveConfig()
    config.set_adapter(Adapter())
    monkeypatch.setattr(checkpoint_patch, "_config", config)

    with patch.dict(
        sys.modules,
        {"safetensors.torch": fake_safetensors},
        clear=False,
    ):
        getattr(checkpoint_patch, "_convert_to_hf_and_save")(
            {
                "model.layers.0.moe.experts.0.w1": expert_tensor,
                "model.layers.0.attn.weight": torch.zeros(2, 2),
            },
            str(tmp_path),
        )

    assert expert_tensor.full_tensor_calls == 1
    saved_tensors = saved.get("tensors")

    assert saved_tensors is not None
    assert sorted(saved_tensors.keys()) == [
        "model.layers.0.attn.weight",
        "model.layers.0.moe.experts.0.w1",
    ]
