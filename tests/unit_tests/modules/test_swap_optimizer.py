# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import types
from unittest.mock import patch

import torch
import torchtitan.components.optimizer as tt_optimizer

from torchtitan_npu.patches.optimizer import swap_optimizer


def test_unwrap_dtensor_returns_plain_tensor_for_non_dtensor():
    tensor = torch.randn(2, 2)

    result = swap_optimizer.unwrap_dtensor(tensor)

    assert result is tensor


def test_build_optimizers_wrapper_delegates_when_swap_disabled():
    sentinel = object()
    calls = []
    with patch.object(
        swap_optimizer,
        "_original_build_optimizers",
        lambda model_parts, optimizer_config, parallel_dims, ft_manager: calls.append(
            (model_parts, optimizer_config, parallel_dims, ft_manager)
        )
        or sentinel,
    ):
        optimizer_config = types.SimpleNamespace(swap_optimizer=False)

        result = tt_optimizer.build_optimizers(
            model_parts=["model"],
            optimizer_config=optimizer_config,
            parallel_dims="parallel_dims",
            ft_manager="ft_manager",
        )

    assert result is sentinel
    assert calls == [(["model"], optimizer_config, "parallel_dims", "ft_manager")]


def test_build_optimizers_wrapper_rejects_unknown_optimizer(monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, closure=None: None)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, closure=None: None)

    optimizer_config = types.SimpleNamespace(
        swap_optimizer=True,
        name="SGD",
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="fused",
        swap_optimizer_times=8,
    )

    try:
        tt_optimizer.build_optimizers(
            model_parts=[],
            optimizer_config=optimizer_config,
            parallel_dims=None,
            ft_manager=None,
        )
        assert False, "Expected NotImplementedError for unsupported optimizer"
    except NotImplementedError as exc:
        assert "Optimizer SGD not added" in str(exc)


def test_build_optimizers_wrapper_uses_swap_container(monkeypatch):
    calls = []

    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, closure=None: None)
    monkeypatch.setattr(torch.optim.Adam, "step", lambda self, closure=None: None)
    monkeypatch.setattr(
        swap_optimizer,
        "SwapOptimizersContainer",
        lambda model_parts, optimizer_cls, optimizer_kwargs, swap_times: calls.append(
            (model_parts, optimizer_cls, optimizer_kwargs, swap_times)
        ) or "swap_container",
    )

    optimizer_config = types.SimpleNamespace(
        swap_optimizer=True,
        name="AdamW",
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.1,
        implementation="fused",
        swap_optimizer_times=16,
    )

    result = tt_optimizer.build_optimizers(
        model_parts=["model_part"],
        optimizer_config=optimizer_config,
        parallel_dims=None,
        ft_manager=None,
    )

    assert result == "swap_container"
    assert calls[0][0] == ["model_part"]
    assert calls[0][1] is torch.optim.AdamW
    assert calls[0][2]["lr"] == 1e-3
    assert calls[0][3] == 16

