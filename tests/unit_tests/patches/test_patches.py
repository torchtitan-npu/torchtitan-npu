# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torchtitan.distributed.activation_checkpoint as upstream_activation_checkpoint

from torchtitan_npu.patches.distributed.custom_context_parallel import (
    CustomContextParallelContext,
)
from torchtitan_npu.patches.quantization import quantize
from torchtitan_npu.patches.torch import clip_grad
from torchtitan_npu.patches.torchtitan import activation_checkpoint


class DummyMesh:
    def __init__(self, size_value):
        self.size_value = size_value

    def size(self):
        return self.size_value


def test_custom_context_parallel_rejects_mismatched_buffer_dims():
    mesh = DummyMesh(2)
    buffer = torch.randn(2, 4)

    try:
        CustomContextParallelContext(
            mesh,
            buffers=[buffer],
            buffer_seq_dims=[],
        )
        raise AssertionError("Expected ValueError for mismatched buffers and dims")
    except ValueError as exc:
        assert "same number of elements" in str(exc)


def test_custom_context_parallel_rejects_unknown_no_restore_buffer():
    mesh = DummyMesh(2)
    buffer = torch.randn(2, 4)
    foreign_buffer = torch.randn(2, 4)

    try:
        CustomContextParallelContext(
            mesh,
            buffers=[buffer],
            buffer_seq_dims=[1],
            no_restore_buffers={foreign_buffer},
        )
        raise AssertionError("Expected ValueError for invalid no_restore_buffers")
    except ValueError as exc:
        assert "subset of `buffers`" in str(exc)


def test_register_quantize_module_handler_registers_handler():
    class DummyConfig:
        pass

    def handler(module, config):
        return module

    handler_registry = {}
    with patch.object(quantize, "_QUANTIZE_CONFIG_HANDLER", handler_registry):
        decorated = quantize.register_quantize_module_handler(DummyConfig)(handler)
        assert decorated is handler
        assert handler_registry.get(DummyConfig) is handler


def test_group_dtensors_by_layout_groups_non_dtensors_together():
    tensor_a = torch.randn(2, 2)
    tensor_b = torch.randn(2, 2)

    grouped = clip_grad.group_dtensors_by_layout([tensor_a, tensor_b])

    assert len(grouped) == 1
    assert ("non_dtensor", None) in grouped
    assert grouped[("non_dtensor", None)] == [tensor_a, tensor_b]


def test_activation_checkpoint_patch_wraps_upstream_apply_full_ac(monkeypatch):
    captured = {}
    module = torch.nn.Linear(2, 2)
    ac_config = SimpleNamespace(
        preserve_rng_state=True,
        determinism_check="default",
        early_stop=True,
        debug=False,
    )

    def fake_checkpoint_wrapper(wrapped_module, **kwargs):
        captured["module"] = wrapped_module
        captured.update(kwargs)
        return "wrapped"

    monkeypatch.setattr(
        activation_checkpoint,
        "ptd_checkpoint_wrapper",
        fake_checkpoint_wrapper,
    )

    result = upstream_activation_checkpoint._apply_full_ac(module, ac_config)
    contexts = captured["context_fn"]()

    assert result == "wrapped"
    assert captured["module"] is module
    assert captured["preserve_rng_state"] is True
    assert captured["determinism_check"] == "default"
    assert captured["early_stop"] is True
    assert captured["debug"] is False
    assert len(contexts) == 2
    assert all(hasattr(ctx, "__enter__") for ctx in contexts)
    assert all(hasattr(ctx, "__exit__") for ctx in contexts)
