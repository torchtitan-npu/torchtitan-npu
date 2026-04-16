# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import patch

import pytest
import torch

from tests.smoke_tests.model_parallel._multi_rank import (
    FourRankMultiRankTestBase,
    mark_multi_rank_nightly,
    MULTI_RANK_AVAILABLE,
    with_comms,
)
from tests.testing.parallel_dims import assert_single_rank_mesh, build_parallel_dims


pytestmark = pytest.mark.smoke


def _build_parallel_dims(tp, ep, world_size=2):
    return build_parallel_dims(tp=tp, ep=ep, world_size=world_size)


def _assert_tp_mesh_partition(device_type):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard

    mesh = init_device_mesh(device_type, (2, 2), mesh_dim_names=("tp", "ep"))
    tp_mesh = mesh["tp"]
    tp_index = mesh.get_coordinate()[0]
    full_tensor = torch.arange(
        32 * 64, device=device_type, dtype=torch.float32
    ).reshape(32, 64)
    distributed = distribute_tensor(full_tensor, tp_mesh, [Shard(0)])
    expected_local = full_tensor.chunk(tp_mesh.size(), dim=0)[tp_index]

    assert tp_mesh.mesh_dim_names == ("tp",)
    assert distributed.to_local().shape == expected_local.shape
    assert torch.equal(distributed.to_local(), expected_local)


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
def test_two_card_tp_ep_parallel_dims_flags():
    parallel_dims = _build_parallel_dims(tp=2, ep=2)

    assert parallel_dims.tp_enabled
    assert parallel_dims.ep_enabled
    assert not parallel_dims.cp_enabled


@patch("torchtitan.distributed.parallel_dims.device_type", "cpu")
@pytest.mark.usefixtures("single_rank_process_group")
def test_single_rank_mesh_build():
    assert_single_rank_mesh(
        _build_parallel_dims(tp=1, ep=1, world_size=1),
        optional_meshes=("tp", "ep"),
    )


if MULTI_RANK_AVAILABLE:

    @mark_multi_rank_nightly
    class TestTpEpMultiRank(FourRankMultiRankTestBase):
        @with_comms
        def test_tp_ep_dtensor_local_shape(self):
            _assert_tp_mesh_partition(self.device_type)
