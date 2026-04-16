# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.distributed import ParallelDims


def build_parallel_dims(
    *,
    dp_replicate=1,
    dp_shard=1,
    cp=1,
    tp=1,
    pp=1,
    ep=1,
    etp=1,
    world_size=1,
):
    return ParallelDims(
        dp_replicate=dp_replicate,
        dp_shard=dp_shard,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=ep,
        etp=etp,
        world_size=world_size,
    )


def assert_optional_meshes_none(parallel_dims, mesh_names):
    for mesh_name in mesh_names:
        if parallel_dims.get_optional_mesh(mesh_name) is not None:
            raise AssertionError(f"Optional mesh {mesh_name} should be None")


def assert_single_rank_mesh(parallel_dims, optional_meshes=()):
    world_mesh = parallel_dims.build_mesh()
    if world_mesh is None:
        raise AssertionError("World mesh should not be None")
    if world_mesh.size() != 1:
        raise AssertionError(
            f"Expected single-rank world mesh, got size={world_mesh.size()}"
        )
    assert_optional_meshes_none(parallel_dims, optional_meshes)
    return world_mesh
