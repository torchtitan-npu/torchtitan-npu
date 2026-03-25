# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import patch

import pytest
from torchtitan.distributed import ParallelDims

from tests.testing.parallel_dims import assert_single_rank_mesh, build_parallel_dims


def _single_rank_parallel_dims():
    return build_parallel_dims()


def test_context_parallel_module_import():
    from torchtitan_npu.distributed.context_parallel import dsa_cp

    assert dsa_cp is not None


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_auto_calculates_dp_shard():
    parallel_dims = ParallelDims(
        dp_replicate=2,
        dp_shard=-1,
        cp=1,
        tp=2,
        pp=1,
        ep=1,
        etp=1,
        world_size=8,
    )

    assert parallel_dims.dp_shard == 2


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_invalid_world_size_raises():
    with pytest.raises(AssertionError):
        ParallelDims(
            dp_replicate=2,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=1,
            etp=1,
            world_size=10,
        )


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_invalid_zero_parallelism_raises():
    with pytest.raises(AssertionError):
        build_parallel_dims(dp_replicate=0)


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_invalid_etp_combination_raises():
    with pytest.raises(AssertionError):
        ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=4,
            pp=1,
            ep=2,
            etp=2,
            world_size=8,
        )


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_enabled_properties_reflect_parallel_config():
    parallel_dims = ParallelDims(
        dp_replicate=2,
        dp_shard=2,
        cp=2,
        tp=4,
        pp=1,
        ep=1,
        etp=1,
        world_size=32,
    )

    assert parallel_dims.dp_enabled
    assert parallel_dims.dp_replicate_enabled
    assert parallel_dims.dp_shard_enabled
    assert parallel_dims.cp_enabled
    assert parallel_dims.tp_enabled
    assert not parallel_dims.pp_enabled
    assert not parallel_dims.ep_enabled
    assert not parallel_dims.etp_enabled
    assert parallel_dims.dp_cp_enabled
    assert parallel_dims.fsdp_enabled


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_derived_parallel_properties():
    parallel_dims = ParallelDims(
        dp_replicate=2,
        dp_shard=3,
        cp=2,
        tp=4,
        pp=1,
        ep=1,
        etp=1,
        world_size=48,
    )

    assert parallel_dims.fsdp_gradient_divide_factor == 12
    assert parallel_dims.non_data_parallel_size == 8
    assert parallel_dims.seq_len_divisor == 16


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_get_optional_mesh_triggers_lazy_build(single_rank_process_group):
    parallel_dims = _single_rank_parallel_dims()

    assert parallel_dims.get_optional_mesh("tp") is None
    assert parallel_dims.world_mesh.size() == 1
    assert parallel_dims.get_all_one_dimensional_meshes() == {}


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_get_mesh_invalid_name_raises(single_rank_process_group):
    parallel_dims = _single_rank_parallel_dims()
    parallel_dims.build_mesh()

    with pytest.raises(ValueError):
        parallel_dims.get_mesh("invalid_mesh")


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_single_rank_mesh_behaviour(single_rank_process_group):
    parallel_dims = _single_rank_parallel_dims()

    world_mesh = assert_single_rank_mesh(parallel_dims, optional_meshes=("tp", "cp"))
    assert parallel_dims.world_mesh.size() == 1
    assert parallel_dims.get_optional_mesh(["dp_replicate", "fsdp"]) is None
    assert parallel_dims.get_all_one_dimensional_meshes() == {}


@pytest.mark.usefixtures("single_rank_process_group")
@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_get_mesh_requires_enabled_dimension():
    parallel_dims = _single_rank_parallel_dims()

    with pytest.raises(ValueError, match="not available"):
        parallel_dims.get_mesh("tp")
