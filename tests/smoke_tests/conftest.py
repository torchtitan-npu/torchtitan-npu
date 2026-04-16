# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed testing utilities for smoke tests.

This module provides utilities for distributed testing following
PyTorch and torchtitan patterns.

Key patterns from PyTorch/torchtitan:
1. DTensorTestBase - Base class for distributed tests
2. @with_comms decorator - Sets up process groups
3. world_size property - Specify number of ranks

Usage:
    # For single-rank tests (no distributed needed)
    def test_something(npu_device):
        model = nn.Linear(64, 64).to(npu_device)
        ...

    # For multi-rank distributed tests
    class TestDistributed(DTensorTestBase):
        @property
        def world_size(self):
            return 4

        @with_comms
        def test_dtensor(self):
            mesh = init_device_mesh("npu", (4,))
            ...
"""

import functools
import os
import unittest
from collections.abc import Callable
from contextlib import contextmanager

import pytest
import torch
import torch.distributed as dist


# ============================================================================
# Distributed Test Base Classes (following PyTorch patterns)
# ============================================================================


class DistributedTestBase(unittest.TestCase):
    """
    Base class for distributed tests following PyTorch patterns.

    This class provides:
    - Automatic distributed environment setup/teardown
    - world_size property for specifying number of ranks
    - device_type property for backend selection

    Similar to torch.testing._internal.distributed._tensor.common_dtensor.DTensorTestBase
    """

    @property
    def world_size(self) -> int:
        """Number of ranks for distributed tests. Override in subclasses."""
        return 1

    @property
    def device_type(self) -> str:
        """Device type for distributed tests. Returns 'npu' or 'cpu'."""
        if hasattr(torch, "npu") and torch.npu.is_available():
            return "npu"
        return "cpu"

    @property
    def backend(self) -> str:
        """Distributed backend to use."""
        if self.device_type == "npu":
            return "nccl"  # NPU uses nccl-like backend
        return "gloo"

    def setUp(self):
        """Set up distributed environment."""
        super().setUp()
        self.ensure_distributed_initialized()

    def tearDown(self):
        """Tear down distributed environment."""
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()

    def ensure_distributed_initialized(self):
        """Initialize distributed communication when it is not ready."""
        if not dist.is_initialized():
            self._init_distributed()

    def _init_distributed(self):
        """Initialize distributed process group."""
        # Use environment variables for distributed init
        # This allows running with torchrun or similar
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            # Single-process testing
            rank = 0
            world_size = 1
            local_rank = 0

        if world_size > 1:
            # For multi-process, use init_method
            os.environ.setdefault("MASTER_ADDR", "localhost")
            os.environ.setdefault("MASTER_PORT", "29500")

            dist.init_process_group(
                backend=self.backend,
                rank=rank,
                world_size=world_size,
            )
            if self.device_type == "npu":
                torch.npu.set_device(local_rank)


def with_comms(func: Callable) -> Callable:
    """
    Decorator to set up distributed communication for tests.

    Similar to torch.testing._internal.distributed._tensor.common_dtensor.with_comms

    This decorator:
    1. Initializes distributed process group before the test
    2. Cleans up after the test

    Usage:
        class MyTest(DistributedTestBase):
            @property
            def world_size(self):
                return 4

            @with_comms
            def test_something(self):
                # Distributed environment is ready
                mesh = init_device_mesh(self.device_type, (self.world_size,))
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # Ensure distributed is initialized
        self.ensure_distributed_initialized()

        try:
            return func(self, *args, **kwargs)
        finally:
            # Clean up if we initialized it
            if dist.is_initialized():
                dist.destroy_process_group()

    return wrapper


# ============================================================================
# Pytest Fixtures for Distributed Testing
# ============================================================================


def distributed_available() -> bool:
    """Check if distributed testing is available."""
    return dist.is_available()


def npu_available() -> bool:
    """Check if NPU is available for testing."""
    return hasattr(torch, "npu") and torch.npu.is_available()


def skip_on_runtime_unsupported(
    error: RuntimeError, unsupported_markers: tuple[str, ...], reason: str
):
    """Skip smoke tests when the current runtime/SOC does not support the requested op shape or dtype."""
    message = str(error)
    if any(marker in message for marker in unsupported_markers):
        pytest.skip(reason)
    raise error


# ============================================================================
# Utility Functions
# ============================================================================


@contextmanager
def distributed_context(
    rank: int,
    world_size: int,
    backend: str = "gloo",
    init_method: str = "tcp://localhost:29500",
):
    """
    Context manager for distributed testing.

    Usage:
        with distributed_context(rank=0, world_size=1) as ctx:
            model = nn.Linear(64, 64)
            # Run distributed code
    """
    try:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
        )
        yield {"rank": rank, "world_size": world_size}
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def get_device_mesh(
    mesh_dims: tuple,
    mesh_dim_names: tuple | None = None,
):
    """
    Create a device mesh for distributed testing.

    Args:
        mesh_dims: Tuple of mesh dimensions, e.g., (2, 4) for 2x4 mesh
        mesh_dim_names: Optional names for each dimension

    Returns:
        DeviceMesh object
    """
    from torch.distributed.device_mesh import init_device_mesh

    device = "npu" if npu_available() else "cpu"

    if mesh_dim_names:
        return init_device_mesh(device, mesh_dims, mesh_dim_names=mesh_dim_names)
    return init_device_mesh(device, mesh_dims)


def create_dtensor(
    tensor: torch.Tensor,
    mesh,
    placements,
):
    """
    Create a distributed tensor from a local tensor.

    Args:
        tensor: Local tensor to distribute
        mesh: DeviceMesh to distribute over
        placements: Placements for distribution (e.g., [Shard(0)])

    Returns:
        DTensor
    """
    from torch.distributed.tensor import distribute_tensor

    return distribute_tensor(tensor, mesh, placements)


# ============================================================================
# Test Helpers
# ============================================================================


def requires_distributed(func: Callable) -> Callable:
    """Decorator to skip tests if distributed is not available."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not distributed_available():
            pytest.skip("Distributed not available")
        return func(*args, **kwargs)

    return wrapper


def requires_npu(func: Callable) -> Callable:
    """Decorator to skip tests if NPU is not available."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not npu_available():
            pytest.skip("NPU not available")
        return func(*args, **kwargs)

    return wrapper


def requires_world_size(min_size: int):
    """
    Decorator factory to skip tests if world_size is less than required.

    Usage:
        @requires_world_size(4)
        def test_tp_4(self):
            # Requires at least 4 GPUs
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.world_size < min_size:
                pytest.skip(f"Requires world_size >= {min_size}, got {self.world_size}")
            return func(self, *args, **kwargs)

        return wrapper

    return decorator
