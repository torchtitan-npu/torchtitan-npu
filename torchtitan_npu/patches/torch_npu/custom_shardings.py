# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
from typing import Callable

import torch
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._op_schema import OpSchema, OpStrategy, TupleStrategy
from torch.distributed.tensor._ops._matrix_ops import _mm_like_strategy


# Access the global sharding propagator registry from PyTorch DTensor
_SHARDING_REGISTRY = DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
aten = torch.ops.aten


def register_sharding_patch(op_overload, patch_fn: Callable[[OpSchema, Callable], OpStrategy]):
    """
    Registers a custom patch for a specific torch operator sharding strategy.

    Args:
        op_overload: The torch operator (e.g., torch.ops.aten.matmul.default).
        patch_fn: A custom function that takes (op_schema, original_handler) and returns OpStrategy.
    """
    if op_overload not in _SHARDING_REGISTRY:
        raise NotImplementedError(f"Op {op_overload} not found in original sharding registry.")

    # Capture the original handler to allow fallback.
    original_handler = _SHARDING_REGISTRY[op_overload]

    @functools.wraps(original_handler)
    def wrapper(op_schema: OpSchema) -> OpStrategy:
        # Delegate logic to the user-defined patch function
        return patch_fn(op_schema, original_handler)

    # Update the global registry with the wrapped handler
    _SHARDING_REGISTRY[op_overload] = wrapper


def matmul_sharding(op_schema: OpSchema, fallback_handler: Callable) -> OpStrategy:
    """
    Custom sharding strategy for NPU matmul.
    """
    mesh = op_schema.get_mesh_from_args()

    # 1. Inspect Input Shapes
    # We only care about the first two arguments (Input and Weight)
    args = op_schema.args_schema
    if len(args) != 2:
        raise NotImplementedError(f"Only support two inputs, current input num is {len(args)}")

    shape1 = args[0].strategies[0].output_spec.shape
    shape2 = args[1].strategies[0].output_spec.shape
    ndim1 = len(shape1)
    ndim2 = len(shape2)

    # 2. Dynamic Einsum Generation
    # Explicitly defining the equation avoids ambiguity for the sharding propagator.
    if ndim1 == 3 and ndim2 == 2:
        if shape1[2] != shape2[0]:
            raise NotImplementedError(f"Input shapes are not 'bmk' and 'kn': {shape1=}, {shape2=}.")
        equation = "bmk,kn->bmn"
        return _mm_like_strategy(equation, mesh, op_schema)

    if ndim1 == 2 and ndim2 == 2:
        if shape1[1] != shape2[0]:
            raise NotImplementedError(f"Input shapes are not 'mk' and 'kn': {shape1=}, {shape2=}.")
        equation = "mk,kn->mn"
        return _mm_like_strategy(equation, mesh, op_schema)

    # 3. Fallback
    # For unsupported shapes (e.g., broadcasting, 4D), revert to original logic.
    return fallback_handler(op_schema)


# apply op sharding patches
register_sharding_patch(aten.matmul.default, matmul_sharding)