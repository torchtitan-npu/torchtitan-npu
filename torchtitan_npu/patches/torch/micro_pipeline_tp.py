# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file is derived from PyTorch,
# https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/utils/clip_grad.py
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
NPU Async-Tp integration for Inductor.

Registers NPU (PrivateUse1) kernels for fused all-gather+matmul and
matmul+reduce-scatter, and installs the corresponding Inductor FX inserters.
"""

from functools import lru_cache

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

import torch_npu
from torch.distributed import _symmetric_memory
from torchtitan.distributed import tensor_parallel as _tp_mod


@lru_cache(maxsize=None)
def _get_hccl_comm_name(group_name: str) -> str | None:
    if not dist.is_initialized():
        return None
    # pyrefly: ignore [bad-argument-type]
    group = c10d._resolve_process_group(group_name)
    if group is None:
        return None
    rank = dist.get_rank(group)
    # pyrefly: ignore [missing-attribute]
    return group._get_backend(torch.device("npu")).get_hccl_comm_name(rank)


def _npu_kernel_fused_all_gather_matmul(
    A_shard: torch.Tensor,
    Bs: list[torch.Tensor],
    gather_dim: int,
    group_name: str,
    *,
    return_A: bool = True,
) -> tuple[torch.Tensor | None, list[torch.Tensor]]:
    if len(Bs) > 1:
        return_A = True
    # pyrefly: ignore [bad-argument-type]
    group = c10d._resolve_process_group(group_name)
    world_size = group.size()
    shard_moved = A_shard.movedim(gather_dim, 0).contiguous()
    leading_dims_before_flat = list(shard_moved.shape[:-1])
    shard_flat = shard_moved.flatten(0, -2)

    def unflatten(t: torch.Tensor) -> torch.Tensor:
        new_shape = [world_size] + leading_dims_before_flat + [t.shape[-1]]
        t = t.view(*new_shape).flatten(0, 1).movedim(0, gather_dim)
        return t

    hcom = _get_hccl_comm_name(group_name)
    outputs = []
    left_gathered = None

    if len(Bs) == 0:
        return None, outputs
    matmul_output, gathered_left = torch_npu.npu_all_gather_base_mm(
        shard_flat,
        Bs[0],
        hcom,
        world_size,
        gather_index=0,
        gather_output=return_A,
    )
    outputs.append(matmul_output)
    left_gathered = gathered_left
    if left_gathered is None and len(Bs) > 1:
        raise RuntimeError(
            "[Async-TP] expected gathered left from npu_all_gather_base_mm for multiple Bs"
        )
    for B in Bs[1:]:
        outputs.append(torch.matmul(left_gathered, B))

    left_out = (
        unflatten(left_gathered) if (return_A and left_gathered is not None) else None
    )
    return left_out, [unflatten(out) for out in outputs]


def _npu_kernel_fused_matmul_reduce_scatter(
    A: torch.Tensor,
    B: torch.Tensor,
    reduce_op: str,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    # pyrefly: ignore [bad-argument-type]
    group = c10d._resolve_process_group(group_name)
    world_size = group.size()

    A_moved = A.movedim(scatter_dim, 0).contiguous()
    leading_dims_before_flat = list(A_moved.shape[:-1])
    A_flat = A_moved.flatten(0, -2)

    hcom = _get_hccl_comm_name(group_name)
    result = torch_npu.npu_mm_reduce_scatter_base(
        A_flat, B, hcom, world_size, reduce_op="sum"
    )

    out_M = leading_dims_before_flat[0] // world_size
    if A.dim() > 2:
        result = result.view(out_M, *leading_dims_before_flat[1:], B.shape[1])
        result = result.movedim(0, scatter_dim)

    return result


_npu_tp_patches_applied = False


"""Register PrivateUse1 kernels for NPU fused all_gather_matmul / matmul_reduce_scatter. Inserters use upstream."""


def install_npu_tp_patches():
    global _npu_tp_patches_applied
    if _npu_tp_patches_applied:
        return
    _npu_tp_patches_applied = True
    torch.library.register_kernel(
        torch.ops.symm_mem.fused_all_gather_matmul.default,
        "PrivateUse1",
        _npu_kernel_fused_all_gather_matmul,
    )
    torch.library.register_kernel(
        torch.ops.symm_mem.fused_matmul_reduce_scatter.default,
        "PrivateUse1",
        _npu_kernel_fused_matmul_reduce_scatter,
    )

    torch.distributed.is_nccl_available = dist.is_nccl_available = lambda: True
    # Force-enable symmetric memory so micro-pipeline TP fusions are always allowed on NPU.
    _symmetric_memory.is_symm_mem_enabled_for_group = lambda group_name: True


_orig_maybe_enable_async_tp = _tp_mod.maybe_enable_async_tp


def _enable_async_tp_wrapper(job_config, tp_mesh):
    _orig_maybe_enable_async_tp(job_config, tp_mesh)
    if getattr(job_config.parallelism, "enable_async_tensor_parallel", False):
        install_npu_tp_patches()


_tp_mod.maybe_enable_async_tp = _enable_async_tp_wrapper
