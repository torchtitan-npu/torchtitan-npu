# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Pytest configuration for torchtitan-npu tests.

This conftest ensures that torchtitan_npu patches are applied before
running any tests, including torchtitan upstream tests.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


def pytest_configure(config):
    """
    Called before test collection and execution.
    Import torchtitan_npu to apply all NPU patches.
    """
    import torchtitan_npu  # noqa: F401


@dataclass
class MuonOptimizerConfig:
    """Typed configuration for Muon optimizer tests."""

    name: str = "Muon"
    lr: float = 1e-3
    weight_decay: float = 0.01
    muon_lr: float | None = None
    muon_momentum: float = 0.95
    muon_enable_nesterov: bool = True
    muon_ns_steps: int = 5
    muon_adjust_lr_fn: str | None = None
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    implementation: str = "for-loop"

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**{k: v for k, v in self.__dict__.items()})


@dataclass
class LRSchedulerTestConfig:
    """Typed configuration for LR scheduler tests."""

    warmup_steps: int = 2
    decay_ratio: float = 0.8
    decay_type: str = "cosine"
    min_lr_factor: float = 0.1

    def to_namespace(self) -> SimpleNamespace:
        return SimpleNamespace(**{k: v for k, v in self.__dict__.items()})


@pytest.fixture(scope="session")
def npu_available():
    """Return whether a real NPU runtime is available."""
    return hasattr(torch, "npu") and torch.npu.is_available()


@pytest.fixture(scope="session")
def npu_device(npu_available):
    """Provide a shared NPU device fixture for smoke tests."""
    if not npu_available:
        pytest.skip("NPU not available")
    return torch.device("npu:0")


@pytest.fixture
def muon_config():
    """Factory fixture for creating Muon optimizer configs."""

    def _make_config(**overrides):
        base = MuonOptimizerConfig()
        for k, v in overrides.items():
            setattr(base, k, v)
        return base.to_namespace()

    return _make_config


@pytest.fixture
def muon_optimizer_config():
    """Factory fixture for creating typed Muon optimizer configs."""

    def _make_config(**overrides):
        return MuonOptimizerConfig(**overrides)

    return _make_config


@pytest.fixture
def lr_scheduler_config():
    """Factory fixture for creating typed LR scheduler configs."""

    def _make_config(**overrides):
        return LRSchedulerTestConfig(**overrides)

    return _make_config


def stable_randn(*shape, device, dtype=torch.float32, scale=0.01, requires_grad=False):
    """Generate small-amplitude random tensors to avoid unstable smoke inputs."""
    tensor = torch.randn(*shape, dtype=torch.float32, device=device) * scale
    tensor = tensor.to(dtype)
    if requires_grad:
        tensor.requires_grad_()
    return tensor


def assert_tensor_finite(value, message="Tensor should be finite"):
    """Check finiteness on CPU to avoid NPU-side isfinite inconsistencies."""
    if not torch.isfinite(value.detach().float().cpu()).all().item():
        raise AssertionError(message)


@pytest.fixture
def single_rank_process_group():
    """Provide a shared single-rank process group for mesh-related tests."""
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://localhost:12356",
            world_size=1,
            rank=0,
        )
    yield
    if dist.is_initialized():
        dist.destroy_process_group()
