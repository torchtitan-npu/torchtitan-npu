# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from torchtitan_npu.patches.optimizer.muon_optimizer import (
    build_muon_hybrid_optimizers,
    MuonHybridOptimizersContainer,
)


def test_params_classification(muon_config):
    """Test parameter classification for Muon vs AdamW.

    Rules:
    - 2D parameters go to Muon, except embeddings, lm_head, and output
    - Non-2D parameters go to AdamW
    - embeddings, lm_head, output are 2D but should go to AdamW
    """

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 16)
            self.linear = nn.Linear(16, 8)
            self.lm_head = nn.Linear(8, 100)
            self.output = nn.Linear(8, 100)
            self.norm = nn.LayerNorm(8)

    model = TestModel()
    config = muon_config()

    optimizer = build_muon_hybrid_optimizers([model], config, None)

    muon = optimizer.optimizers[0]
    adamw = optimizer.optimizers[1]

    muon_params = set(muon.param_groups[0]["params"])
    adamw_params = set(adamw.param_groups[0]["params"])

    assert model.linear.weight in muon_params
    assert model.linear.weight not in adamw_params

    assert model.linear.bias not in muon_params
    assert model.linear.bias in adamw_params

    assert model.norm.weight not in muon_params
    assert model.norm.weight in adamw_params

    assert model.embed.weight not in muon_params
    assert model.embed.weight in adamw_params

    assert model.lm_head.weight not in muon_params
    assert model.lm_head.weight in adamw_params

    assert model.output.weight not in muon_params
    assert model.output.weight in adamw_params


def test_build_hybrid_creates_muon_and_adamw(muon_config):
    model = nn.Linear(8, 8)
    config = muon_config()

    optimizer = build_muon_hybrid_optimizers([model], config, None)

    assert isinstance(optimizer, MuonHybridOptimizersContainer)
    assert len(optimizer.optimizers) == 2

    optim_types = [type(o).__name__ for o in optimizer.optimizers]
    assert "Muon" in optim_types
    assert "AdamW" in optim_types


def test_build_hybrid_config_kwargs(muon_config):
    config = muon_config(
        lr=0.05,
        muon_lr=0.02,
        muon_momentum=0.99,
        muon_enable_nesterov=False,
        muon_adjust_lr_fn="original",
        beta1=0.8,
        beta2=0.9,
    )
    model = nn.Linear(8, 8)

    optimizer = build_muon_hybrid_optimizers([model], config, None)

    muon = optimizer.optimizers[0]
    adamw = optimizer.optimizers[1]

    assert muon.param_groups[0]["lr"] == 0.02
    assert muon.param_groups[0]["momentum"] == 0.99
    assert muon.param_groups[0]["nesterov"] is False

    assert adamw.param_groups[0]["lr"] == 0.05
    assert adamw.param_groups[0]["betas"] == (0.8, 0.9)


def test_muon_lr_ignored_with_match_rms_adamw(muon_config):
    config = muon_config(
        lr=0.05,
        muon_lr=0.02,
        muon_adjust_lr_fn="match_rms_adamw",
    )
    model = nn.Linear(8, 8)

    optimizer = build_muon_hybrid_optimizers([model], config, None)

    muon = optimizer.optimizers[0]

    assert muon.param_groups[0]["lr"] == 0.05


def test_invalid_implementation_raises(muon_config):
    config = muon_config(implementation="invalid")
    model = nn.Linear(8, 8)

    with pytest.raises(ValueError, match="Invalid implementation"):
        build_muon_hybrid_optimizers([model], config, None)


def test_container_step_calls_all_optimizers():
    calls = []

    def mock_step():
        calls.append("step")

    model = nn.Linear(8, 8)
    optimizer = MuonHybridOptimizersContainer(
        [model],
        [
            MagicMock(step=mock_step),
            MagicMock(step=mock_step),
        ],
    )

    optimizer.step()

    assert len(calls) == 2


def test_container_zero_grad_calls_all_optimizers():
    calls = []

    def mock_zero_grad(set_to_none=True):
        calls.append(set_to_none)

    model = nn.Linear(8, 8)
    optimizer = MuonHybridOptimizersContainer(
        [model],
        [
            MagicMock(zero_grad=mock_zero_grad),
            MagicMock(zero_grad=mock_zero_grad),
        ],
    )

    optimizer.zero_grad(set_to_none=True)

    assert calls == [True, True]


def test_underlying_optimizer_state_exists(muon_config):
    """Test that underlying optimizers have state after training."""
    model = nn.Linear(8, 8)
    config = muon_config(muon_momentum=0.9)

    optimizer = build_muon_hybrid_optimizers([model], config, None)

    for _ in range(3):
        optimizer.zero_grad()
        loss = model(torch.randn(2, 8)).sum()
        loss.backward()
        optimizer.step()

    muon = optimizer.muon_optimizer
    adamw = optimizer.adamw_optimizer

    muon_state = muon.state_dict()
    adamw_state = adamw.state_dict()

    assert len(muon_state["state"]) > 0, "Muon should have optimizer state"
    assert len(adamw_state["state"]) > 0, "AdamW should have optimizer state"


def test_optimizer_state_persistence_muon_momentum(muon_config):
    """Test Muon momentum buffer survives checkpoint save/load."""
    config = muon_config(muon_momentum=0.9, muon_adjust_lr_fn="original")
    model = nn.Linear(8, 8)
    optimizer = build_muon_hybrid_optimizers([model], config, None)

    for _ in range(5):
        optimizer.zero_grad()
        loss = model(torch.randn(2, 8)).sum()
        loss.backward()
        optimizer.step()

    muon = optimizer.muon_optimizer
    muon_state_before = muon.state_dict()
    momentum_before = muon_state_before["state"]

    assert len(momentum_before) > 0, "Muon should have momentum state"
    for _, param_state in momentum_before.items():
        if "momentum_buffer" in param_state:
            momentum_buf = param_state["momentum_buffer"]
            assert not torch.allclose(
                momentum_buf, torch.zeros_like(momentum_buf)
            ), "Momentum buffer should be non-zero after training"

    model2 = nn.Linear(8, 8)
    optimizer2 = build_muon_hybrid_optimizers([model2], config, None)

    muon2 = optimizer2.muon_optimizer
    muon2.load_state_dict(muon_state_before)

    muon_state_after = muon2.state_dict()
    momentum_after = muon_state_after["state"]

    for param_id in momentum_before:
        if "momentum_buffer" in momentum_before[param_id]:
            buf_before = momentum_before[param_id]["momentum_buffer"]
            buf_after = momentum_after[param_id]["momentum_buffer"]
            assert torch.allclose(
                buf_before, buf_after, atol=1e-6
            ), "Muon momentum buffer should be preserved after checkpoint load"


def test_optimizer_state_persistence_adamw_exp_avg(muon_config):
    """Test AdamW exp_avg and exp_avg_sq survive checkpoint save/load."""
    config = muon_config(beta1=0.9, beta2=0.95, muon_adjust_lr_fn="original")
    model = nn.Linear(8, 8)
    optimizer = build_muon_hybrid_optimizers([model], config, None)

    for _ in range(5):
        optimizer.zero_grad()
        loss = model(torch.randn(2, 8)).sum()
        loss.backward()
        optimizer.step()

    adamw = optimizer.adamw_optimizer
    adamw_state_before = adamw.state_dict()
    state_before = adamw_state_before["state"]

    assert len(state_before) > 0, "AdamW should have optimizer state"
    for _, param_state in state_before.items():
        assert "exp_avg" in param_state, "AdamW should have exp_avg"
        assert "exp_avg_sq" in param_state, "AdamW should have exp_avg_sq"

        exp_avg = param_state["exp_avg"]
        exp_avg_sq = param_state["exp_avg_sq"]

        assert not torch.allclose(
            exp_avg, torch.zeros_like(exp_avg), atol=1e-10
        ), "exp_avg should be non-zero after training"
        assert not torch.allclose(
            exp_avg_sq, torch.zeros_like(exp_avg_sq), atol=1e-10
        ), "exp_avg_sq should be non-zero after training"

    model2 = nn.Linear(8, 8)
    optimizer2 = build_muon_hybrid_optimizers([model2], config, None)

    adamw2 = optimizer2.adamw_optimizer
    adamw2.load_state_dict(adamw_state_before)

    adamw_state_after = adamw2.state_dict()
    state_after = adamw_state_after["state"]

    for param_id in state_before:
        exp_avg_before = state_before[param_id]["exp_avg"]
        exp_avg_after = state_after[param_id]["exp_avg"]
        assert torch.allclose(
            exp_avg_before, exp_avg_after, atol=1e-6
        ), "AdamW exp_avg should be preserved after checkpoint load"

        exp_avg_sq_before = state_before[param_id]["exp_avg_sq"]
        exp_avg_sq_after = state_after[param_id]["exp_avg_sq"]
        assert torch.allclose(
            exp_avg_sq_before, exp_avg_sq_after, atol=1e-6
        ), "AdamW exp_avg_sq should be preserved after checkpoint load"


def test_multiple_model_parts_param_distribution(muon_config):
    """Test parameter distribution across optimizers with multiple model parts."""
    model1 = nn.Linear(8, 8)
    model2 = nn.Linear(8, 8)
    config = muon_config()

    optimizer = build_muon_hybrid_optimizers([model1, model2], config, None)

    muon_params = set(optimizer.muon_optimizer.param_groups[0]["params"])
    adamw_params = set(optimizer.adamw_optimizer.param_groups[0]["params"])

    assert model1.weight in muon_params, "model1.weight should be in Muon"
    assert model2.weight in muon_params, "model2.weight should be in Muon"
    assert model1.bias in adamw_params, "model1.bias should be in AdamW"
    assert model2.bias in adamw_params, "model2.bias should be in AdamW"

    all_params = set()
    for model in [model1, model2]:
        all_params.update(p for p in model.parameters() if p.requires_grad)

    assert muon_params | adamw_params == all_params, "All params should be covered"
    assert len(muon_params & adamw_params) == 0, "No overlap between optimizers"


def test_multiple_model_parts_optimizer_state(muon_config):
    """Test optimizer state is built correctly with multiple model parts."""
    model1 = nn.Linear(8, 8)
    model2 = nn.Linear(8, 8)
    config = muon_config()

    optimizer = build_muon_hybrid_optimizers([model1, model2], config, None)

    for _ in range(3):
        optimizer.zero_grad()
        loss = model1(torch.randn(2, 8)).sum() + model2(torch.randn(2, 8)).sum()
        loss.backward()
        optimizer.step()

    muon_state = optimizer.muon_optimizer.state_dict()
    adamw_state = optimizer.adamw_optimizer.state_dict()

    muon_param_count = len(optimizer.muon_optimizer.param_groups[0]["params"])
    adamw_param_count = len(optimizer.adamw_optimizer.param_groups[0]["params"])

    assert (
        len(muon_state["state"]) == muon_param_count
    ), f"Muon state should have {muon_param_count} entries"
    assert (
        len(adamw_state["state"]) == adamw_param_count
    ), f"AdamW state should have {adamw_param_count} entries"
