# Adapted from
# https://github.com/pytorch/torchtitan/blob/v0.2.1/torchtitan/distributed/utils.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Add DTensor type check for `non_ep_grads_total_norm` before calling `.full_tensor()`.
Prevents an AttributeError when `non_ep_grads_total_norm` is empty or a plain tensor.
"""

import math
from collections.abc import Iterable

import torch

import torchtitan
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


@torch.no_grad()
def _clip_grad_norm_with_ep(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
) -> torch.Tensor:
    ep_params = []
    non_ep_params = []
    ep_grads = []
    non_ep_grads = []

    for p in parameters:
        if p.grad is None:
            continue
        assert isinstance(p, DTensor) and isinstance(p.grad, DTensor)
        if "ep" in p.device_mesh.mesh_dim_names:
            ep_params.append(p)
            ep_grads.append(p.grad)
        else:
            non_ep_params.append(p)
            non_ep_grads.append(p.grad)
    ep_grads_total_norm = torch.nn.utils.get_total_norm(
        ep_grads, norm_type, error_if_nonfinite, foreach
    )
    # ep_grads may be an empty list, in which case get_total_norm returns tensor(0.), a non-DTensor
    # This can occur in PP + EP setups where certain PP ranks only own non-EP layers, for instance.
    if isinstance(ep_grads_total_norm, DTensor):
        ep_grads_total_norm = ep_grads_total_norm.full_tensor()

    non_ep_grads_total_norm = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type, error_if_nonfinite, foreach
    )
    if isinstance(non_ep_grads_total_norm, DTensor):
        non_ep_grads_total_norm = non_ep_grads_total_norm.full_tensor()

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
    else:
        total_norm = (
            ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        )
        total_norm **= 1.0 / norm_type

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach)

    return total_norm


torchtitan.distributed.utils._clip_grad_norm_with_ep = _clip_grad_norm_with_ep
