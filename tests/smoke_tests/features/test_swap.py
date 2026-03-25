# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torchtitan.components.optimizer as tt_optimizer

from torchtitan_npu.patches.optimizer.swap_optimizer import (
    swap_optimizer_step,
    SwapOptimizersContainer,
)

pytestmark = pytest.mark.smoke


@pytest.fixture(autouse=True)
def _reset_swap_global_state():
    SwapOptimizersContainer.param_to_cpu_states_map.clear()
    SwapOptimizersContainer.param_to_device_states_map.clear()
    SwapOptimizersContainer.swap_to_host_events_map.clear()
    SwapOptimizersContainer.swap_to_device_events_map.clear()
    SwapOptimizersContainer.param_update_events_map.clear()
    yield
    SwapOptimizersContainer.param_to_cpu_states_map.clear()
    SwapOptimizersContainer.param_to_device_states_map.clear()
    SwapOptimizersContainer.swap_to_host_events_map.clear()
    SwapOptimizersContainer.swap_to_device_events_map.clear()
    SwapOptimizersContainer.param_update_events_map.clear()


def _swap_optimizer_config():
    return SimpleNamespace(
        swap_optimizer=True,
        name="AdamW",
        lr=1e-3,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        weight_decay=0.01,
        implementation="fused",
        swap_optimizer_times=2,
    )


def test_swap_optimizer_builds_swap_container(npu_device):
    model = nn.Linear(32, 32).to(npu_device)

    container = tt_optimizer.build_optimizers(
        [model],
        _swap_optimizer_config(),
        parallel_dims=object(),
    )
    optimizer = container.optimizers[0]

    assert isinstance(container, SwapOptimizersContainer)
    assert len(container.optimizers) == 1
    assert optimizer.step.__func__.__name__ == swap_optimizer_step.__name__


def test_swap_optimizer_initializes_cpu_state_buffers(npu_device):
    model = nn.Linear(16, 16).to(npu_device)

    container = tt_optimizer.build_optimizers(
        [model],
        _swap_optimizer_config(),
        parallel_dims=object(),
    )
    optimizer = container.optimizers[0]

    for param in optimizer.param_groups[0]["params"]:
        cpu_state = SwapOptimizersContainer.param_to_cpu_states_map[param]
        assert cpu_state["exp_avg"].device.type == "cpu"
        assert cpu_state["exp_avg_sq"].device.type == "cpu"


def test_swap_optimizer_step_updates_model_parameters(npu_device):
    model = nn.Linear(8, 8).to(npu_device)
    container = tt_optimizer.build_optimizers(
        [model],
        _swap_optimizer_config(),
        parallel_dims=object(),
    )
    optimizer = container.optimizers[0]
    tracked_param = optimizer.param_groups[0]["params"][0]
    baseline_state = SwapOptimizersContainer.param_to_cpu_states_map[tracked_param][
        "exp_avg"
    ].clone()

    x = torch.randn(4, 8, device=npu_device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()

    assert "step" in optimizer.param_groups[0]
    updated_state = SwapOptimizersContainer.param_to_cpu_states_map[tracked_param][
        "exp_avg"
    ]
    assert torch.count_nonzero(updated_state).item() > 0
    assert not torch.equal(updated_state, baseline_state)
