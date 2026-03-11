# Adapted from
# https://github.com/pytorch/pytorch/blob/v2.10.0/torch/nn/utils/clip_grad.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates
# Developed by Huawei Technologies Co., Ltd. based on Meta Platforms, Inc. and affiliates TorchTitan

"""
Optimizer `_get_total_norm` for DTensors.
For finite p-norm, compute gradient norms via local tensors with explicit distributed reductions
instead of DTensor ops, for better performance on NPU.
Infinity norm falls back to the original implementation.
"""

from collections import defaultdict
import math

import torch
import torch.distributed as dist

from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.nn.utils.clip_grad import _no_grad, _tensor_or_tensors
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

_REDUCE_PLACEMENT_NAMES = frozenset({
    'Shard', '_StridedShard', '_Partial', 'Partial'
})


def group_dtensors_by_layout(dtensors):
    """
    Group DTensors by (device_mesh, placements).
    Returns: dict[key -> list[DTensor]]
    """
    groups = defaultdict(list)
    for dt in dtensors:
        if not isinstance(dt, DTensor):
            key = ("non_dtensor", None, None)
        else:
            key = (dt.device_mesh, tuple(dt.placements))
        groups[key].append(dt)
    return groups


def reduce_across_mesh(local_sum, mesh, placements):
    for mesh_dim, placement in enumerate(placements):
        needs_reduce = (
            hasattr(placement, 'reduce_op') or
            type(placement).__name__ in _REDUCE_PLACEMENT_NAMES
        )
        if not needs_reduce:
            continue
        pg = mesh.get_group(mesh_dim=mesh_dim)
        if pg is not None and pg.size() > 1:
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=pg)


def custom_total_norm(norms, norm_type, first_device):
    """
    Translate the DTensor to local_tensor to compute total_norm for acceleration, 
    only supports norm type != inf
    """
    groups = group_dtensors_by_layout(norms)
    global_total_sum = 0.0

    for (mesh, placements), tensors in groups.items():
        group_local_sum = torch.tensor(0.0, device=first_device, dtype=torch.float32)
        for t in tensors:
            t = t.to_local()
            group_local_sum += torch.linalg.vector_norm(t, ord=norm_type, dtype=torch.float32) ** norm_type
        reduce_across_mesh(group_local_sum, mesh, placements)
        global_total_sum += group_local_sum
    
    total_norm = global_total_sum ** (1.0 / norm_type)
    return total_norm


@_no_grad
def _get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: dict[
        tuple[torch.device, torch.dtype], tuple[list[list[Tensor]], list[int]]
    ] = _group_tensors_by_device_and_dtype(  # pyrefly: ignore [bad-assignment]
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: list[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(
                f"foreach=True was passed, but can't use the foreach API on {device.type} tensors"
            )
        else:
            norms.extend(
                [torch.linalg.vector_norm(g, norm_type) for g in device_tensors]
            )

    # Fallback to native execution for infinity norm.
    # The custom distributed reduction logic below assumes finite p-norms.
    # 1. Avoid overflow: The logic below uses `** norm_type` which crashes on `inf`.
    # 2. Avoid logic error: The logic below uses `SUM` reduction, but `L_inf` requires `MAX`.
    if math.isinf(norm_type):
        total_norm = torch.linalg.vector_norm(
            torch.stack([norm.to(first_device) for norm in norms]), norm_type
        )
    else:
        total_norm = custom_total_norm(norms, norm_type, first_device)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm

torch.nn.utils.get_total_norm = _get_total_norm