# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Pytest configuration for torchtitan-npu tests.

This conftest ensures that torchtitan_npu patches are applied before
running any tests, including torchtitan upstream tests.
"""

import pytest
import torch
import torch.distributed as dist


def pytest_configure(config):
    """
    Called before test collection and execution.
    Import torchtitan_npu to apply all NPU patches.
    """
    import torchtitan_npu  # noqa: F401


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
