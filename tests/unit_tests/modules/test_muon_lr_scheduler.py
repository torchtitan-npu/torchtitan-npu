# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from torchtitan.components.lr_scheduler import LRSchedulersContainer

from torchtitan_npu.patches.optimizer.muon_optimizer import (
    build_muon_hybrid_optimizers,
    build_muon_lr_schedulers,
    MuonLRSchedulersContainer,
)


def test_creates_two_independent_schedulers(muon_optimizer_config, lr_scheduler_config):
    """Verify that MuonLRSchedulersContainer creates two LambdaLR schedulers."""
    model = nn.Linear(8, 8)
    opt_config = muon_optimizer_config(muon_adjust_lr_fn="original").to_namespace()
    optimizers = build_muon_hybrid_optimizers([model], opt_config, None)

    lr_config = lr_scheduler_config().to_namespace()
    training_steps = 10

    schedulers = build_muon_lr_schedulers(optimizers, lr_config, training_steps)

    assert isinstance(schedulers, MuonLRSchedulersContainer)
    assert len(schedulers.schedulers) == 2
    assert isinstance(schedulers.schedulers[0], LambdaLR)
    assert isinstance(schedulers.schedulers[1], LambdaLR)


def test_muon_and_adamw_have_different_base_lr(muon_optimizer_config):
    """Verify Muon and AdamW have different base_lr when muon_adjust_lr_fn='original'."""
    model = nn.Linear(8, 8)
    opt_config = muon_optimizer_config(
        lr=2.2e-4, muon_lr=1e-2, muon_adjust_lr_fn="original"
    ).to_namespace()
    optimizers = build_muon_hybrid_optimizers([model], opt_config, None)

    muon_lr = optimizers.muon_optimizer.param_groups[0]["lr"]
    adamw_lr = optimizers.adamw_optimizer.param_groups[0]["lr"]

    assert muon_lr == 1e-2, f"Muon lr should be 1e-2, got {muon_lr}"
    assert adamw_lr == 2.2e-4, f"AdamW lr should be 2.2e-4, got {adamw_lr}"
    assert muon_lr != adamw_lr, "Muon and AdamW should have different base_lr"


def test_step_updates_both_schedulers(muon_optimizer_config):
    """Verify that step() updates both schedulers."""
    model = nn.Linear(8, 8)
    opt_config = muon_optimizer_config().to_namespace()
    optimizers = build_muon_hybrid_optimizers([model], opt_config, None)

    schedulers = MuonLRSchedulersContainer(
        optimizers,
        lr_lambda=lambda step: 1.0,
    )

    initial_epochs = [s.last_epoch for s in schedulers.schedulers]

    schedulers.step()

    for i, s in enumerate(schedulers.schedulers):
        assert (
            s.last_epoch == initial_epochs[i] + 1
        ), f"Scheduler {i} should have incremented last_epoch"


def test_state_dict_saves_first_scheduler_only(muon_optimizer_config):
    """Verify state_dict only saves first scheduler's state (last_epoch)."""
    model = nn.Linear(8, 8)
    opt_config = muon_optimizer_config().to_namespace()
    optimizers = build_muon_hybrid_optimizers([model], opt_config, None)

    schedulers = MuonLRSchedulersContainer(
        optimizers,
        lr_lambda=lambda step: 1.0,
    )

    for _ in range(5):
        schedulers.step()

    state = schedulers.state_dict()

    assert "last_epoch" in state
    assert state["last_epoch"] == 5


def test_load_state_dict_applies_to_both_schedulers(muon_optimizer_config):
    """Verify load_state_dict applies same last_epoch to both schedulers."""
    model = nn.Linear(8, 8)
    opt_config = muon_optimizer_config().to_namespace()
    optimizers = build_muon_hybrid_optimizers([model], opt_config, None)

    schedulers = MuonLRSchedulersContainer(
        optimizers,
        lr_lambda=lambda step: 1.0,
    )

    state = {"last_epoch": 10}

    schedulers.load_state_dict(state)

    assert schedulers.schedulers[0].last_epoch == 10
    assert schedulers.schedulers[1].last_epoch == 10


def test_checkpoint_preserves_independent_base_lr(
    muon_optimizer_config, lr_scheduler_config
):
    """
    Core test: Verify checkpoint save/load preserves independent base_lr.

    This tests the fix for the original issue where AdamW's lr was
    incorrectly replaced by Muon's lr after checkpoint load.
    """
    model = nn.Linear(8, 8)
    opt_config = muon_optimizer_config(
        lr=2.2e-4,
        muon_lr=1e-2,
        muon_adjust_lr_fn="original",
    ).to_namespace()
    optimizers = build_muon_hybrid_optimizers([model], opt_config, None)

    lr_config = lr_scheduler_config(warmup_steps=2, decay_ratio=0.8).to_namespace()
    training_steps = 10

    schedulers = build_muon_lr_schedulers(optimizers, lr_config, training_steps)

    muon_scheduler = schedulers.schedulers[0]
    adamw_scheduler = schedulers.schedulers[1]

    initial_muon_base_lr = muon_scheduler.base_lrs[0]
    initial_adamw_base_lr = adamw_scheduler.base_lrs[0]

    assert initial_muon_base_lr == 1e-2
    assert initial_adamw_base_lr == 2.2e-4

    for _ in range(6):
        schedulers.step()

    saved_state = schedulers.state_dict()

    model2 = nn.Linear(8, 8)
    opt_config2 = muon_optimizer_config(
        lr=2.2e-4,
        muon_lr=1e-2,
        muon_adjust_lr_fn="original",
    ).to_namespace()
    optimizers2 = build_muon_hybrid_optimizers([model2], opt_config2, None)
    schedulers2 = build_muon_lr_schedulers(optimizers2, lr_config, training_steps)

    schedulers2.load_state_dict(saved_state)

    muon_scheduler2 = schedulers2.schedulers[0]
    adamw_scheduler2 = schedulers2.schedulers[1]

    assert (
        muon_scheduler2.base_lrs[0] == initial_muon_base_lr
    ), f"Muon base_lr not preserved: {muon_scheduler2.base_lrs[0]} != {initial_muon_base_lr}"
    assert (
        adamw_scheduler2.base_lrs[0] == initial_adamw_base_lr
    ), f"AdamW base_lr not preserved: {adamw_scheduler2.base_lrs[0]} != {initial_adamw_base_lr}"

    assert (
        schedulers2.schedulers[0].last_epoch == 6
    ), f"Muon scheduler last_epoch should be 6, got {schedulers2.schedulers[0].last_epoch}"
    assert (
        schedulers2.schedulers[1].last_epoch == 6
    ), f"AdamW scheduler last_epoch should be 6, got {schedulers2.schedulers[1].last_epoch}"


def test_match_rms_adamw_uses_standard_scheduler(
    muon_optimizer_config, lr_scheduler_config
):
    """Verify match_rms_adamw mode uses standard LRSchedulersContainer."""
    model = nn.Linear(8, 8)
    opt_config = muon_optimizer_config(
        muon_adjust_lr_fn="match_rms_adamw"
    ).to_namespace()
    optimizers = build_muon_hybrid_optimizers([model], opt_config, None)

    lr_config = lr_scheduler_config().to_namespace()
    training_steps = 10

    schedulers = build_muon_lr_schedulers(optimizers, lr_config, training_steps)

    assert isinstance(
        schedulers, LRSchedulersContainer
    ), f"match_rms_adamw should use standard LRSchedulersContainer, got {type(schedulers)}"
