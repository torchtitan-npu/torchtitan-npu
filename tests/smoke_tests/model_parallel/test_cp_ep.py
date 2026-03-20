# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Smoke tests for model parallelism on the default 2-card smoke environment.

These tests verify dtensor handling under configurations that fit daily smoke.
Requires NPU environment to run.

Run with:
    pytest -v tests/smoke_tests/model_parallel/ -m smoke

Distributed Testing Patterns (following PyTorch/torchtitan):
1. Single-rank tests: Use fixtures, no distributed setup needed
2. Multi-rank nightly tests: Use DTensorTestBase + @with_comms decorator
3. Run multi-rank tests with: torchrun --nproc_per_node=4 pytest tests/...
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from tests.conftest import assert_tensor_finite
from tests.smoke_tests.model_parallel._multi_rank import (
    MULTI_RANK_AVAILABLE,
    FourRankMultiRankTestBase,
    mark_multi_rank_nightly,
    with_comms,
)
from tests.testing.parallel_dims import (
    assert_optional_meshes_none,
    assert_single_rank_mesh,
    build_parallel_dims,
)

# Mark all tests in this module as smoke tests
pytestmark = pytest.mark.smoke


def _build_parallel_dims(*, cp=1, tp=1, ep=1, world_size=1):
    return build_parallel_dims(cp=cp, tp=tp, ep=ep, world_size=world_size)


def _assert_mesh_partition(device_type, mesh_dim_names, shard_dim_name, tensor_shape):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    mesh = init_device_mesh(device_type, (2, 2), mesh_dim_names=mesh_dim_names)
    shard_mesh = mesh[shard_dim_name]
    shard_axis = mesh_dim_names.index(shard_dim_name)
    shard_index = mesh.get_coordinate()[shard_axis]
    tensor = torch.arange(
        tensor_shape[0] * tensor_shape[1],
        device=device_type,
        dtype=torch.float32,
    ).reshape(*tensor_shape)
    distributed = distribute_tensor(tensor, shard_mesh, [Shard(0)])
    expected_local = tensor.chunk(shard_mesh.size(), dim=0)[shard_index]

    assert shard_mesh.mesh_dim_names == (shard_dim_name,)
    assert distributed.to_local().shape == expected_local.shape
    assert torch.equal(distributed.to_local(), expected_local)


def test_npu_tensor_basic(npu_device):
    # Create tensor on NPU
    x = torch.zeros(16, 32, dtype=torch.bfloat16, device=npu_device)

    # Verify tensor properties
    assert x.device.type == "npu", "Tensor should be on NPU"
    assert x.shape == (16, 32), "Tensor shape mismatch"

    # Test tensor operations
    y = x + x
    assert y.shape == x.shape
    assert_tensor_finite(y)


def test_gradient_computation(npu_device):
    # Create simple model
    model = nn.Linear(32, 32).to(npu_device)

    # Forward and backward
    x = torch.randn(4, 32, device=npu_device)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Verify gradients exist and are on NPU
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None, "Gradient should exist"
            assert param.grad.device.type == "npu", "Gradient should be on NPU"


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_valid_two_card_cp_ep_config():
    parallel_dims = _build_parallel_dims(cp=2, ep=2, world_size=2)

    assert parallel_dims.cp == 2
    assert parallel_dims.cp_enabled
    assert parallel_dims.ep_enabled


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_valid_two_card_tp_ep_config():
    parallel_dims = _build_parallel_dims(tp=2, ep=2, world_size=2)

    assert parallel_dims.tp == 2
    assert parallel_dims.tp_enabled
    assert parallel_dims.ep_enabled


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_ep_does_not_change_world_size_requirement():
    parallel_dims = _build_parallel_dims(cp=2, ep=2, world_size=2)

    assert parallel_dims.world_size == 2
    assert parallel_dims.non_data_parallel_size == 2
    assert parallel_dims.ep_enabled


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_invalid_zero_parallelism():
    from torchtitan.distributed import ParallelDims

    with pytest.raises(AssertionError):
        ParallelDims(
            dp_replicate=0,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=1,
        )


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
@pytest.mark.usefixtures("single_rank_process_group")
def test_single_rank_mesh_build():
    assert_single_rank_mesh(_build_parallel_dims())


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
@pytest.mark.usefixtures("single_rank_process_group")
def test_get_optional_mesh_single_rank():
    assert_optional_meshes_none(_build_parallel_dims(), ("tp", "cp", "ep"))


# For multi-rank tests, use the DTensorTestBase pattern
# These tests require running with torchrun:
# torchrun --nproc_per_node=4 -m pytest tests/smoke_tests/model_parallel/test_cp_ep.py -v

if MULTI_RANK_AVAILABLE:
    @mark_multi_rank_nightly
    class TestCpEpMultiRank(FourRankMultiRankTestBase):
        """Test CP+EP operations with multiple ranks.

        Run with: torchrun --nproc_per_node=4 -m pytest ... -k TestCpEpMultiRank
        """

        @with_comms
        def test_dtensor_distribution_cp_ep(self):
            _assert_mesh_partition(self.device_type, ("cp", "ep"), "cp", (16, 32))

        @with_comms
        def test_dtensor_distribution_tp_ep(self):
            _assert_mesh_partition(self.device_type, ("tp", "ep"), "tp", (32, 64))


def test_dsa_cp_context(npu_device):
    try:
        from torchtitan_npu.distributed.context_parallel.dsa_cp import (
            AscendDSAContextParallelContext
        )

        assert AscendDSAContextParallelContext is not None

    except ImportError:
        pytest.skip("DSA CP module not available")


def test_memory_efficiency_parallel(npu_device):
    # Create model
    model = nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    ).to(npu_device)

    # Record initial memory
    torch.npu.empty_cache()
    initial_mem = torch.npu.memory_allocated() / 1024**2

    # Forward pass
    x = torch.randn(8, 256, device=npu_device)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Record peak memory
    peak_mem = torch.npu.max_memory_allocated() / 1024**2

    # Verify memory usage
    assert peak_mem > initial_mem, "Memory should increase during forward/backward"


def test_valid_parallel_config():
    try:
        from torchtitan_npu.config.custom_config import Parallelism

        valid_configs = [
            {"enable_custom_context_parallel": False},
            {"enable_custom_context_parallel": True, "custom_context_parallel_path": "module.Class"},
        ]

        for config in valid_configs:
            parallelism = Parallelism(**config)
            assert parallelism is not None

    except ImportError:
        pytest.skip("Custom config module not available")


