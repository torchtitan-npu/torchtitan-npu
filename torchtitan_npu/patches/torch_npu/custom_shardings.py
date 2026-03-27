# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import functools
from collections.abc import Callable
from typing import cast

import torch
from torch.distributed._tensor.placement_types import DTensorSpec
from torch.distributed.tensor import DeviceMesh, DTensor
from torch.distributed.tensor._op_schema import OpSchema, OpSpec, OpStrategy
from torch.distributed.tensor._ops._matrix_ops import _mm_like_strategy


# Access the global sharding propagator registry from PyTorch DTensor
_SHARDING_REGISTRY = DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
aten = torch.ops.aten


def register_sharding_patch(
    op_overload, patch_fn: Callable[[OpSchema, Callable], OpStrategy]
):
    """
    Registers a custom patch for a specific torch operator sharding strategy.

    Args:
        op_overload: The torch operator (e.g., torch.ops.aten.matmul.default).
        patch_fn: A custom function that takes (op_schema, original_handler) and returns OpStrategy.
    """
    if op_overload not in _SHARDING_REGISTRY:
        raise NotImplementedError(
            f"Op {op_overload} not found in original sharding registry."
        )

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
        raise NotImplementedError(
            f"Only support two inputs, current input num is {len(args)}"
        )

    # pyrefly: ignore [missing-attribute]
    shape1 = args[0].strategies[0].output_spec.shape
    # pyrefly: ignore [missing-attribute]
    shape2 = args[1].strategies[0].output_spec.shape
    ndim1 = len(shape1)
    ndim2 = len(shape2)

    # 2. Dynamic Einsum Generation
    # Explicitly defining the equation avoids ambiguity for the sharding propagator.
    if ndim1 == 3 and ndim2 == 2:
        if shape1[2] != shape2[0]:
            raise NotImplementedError(
                f"Input shapes are not 'bmk' and 'kn': {shape1=}, {shape2=}."
            )
        equation = "bmk,kn->bmn"
        return _mm_like_strategy(equation, mesh, op_schema)

    if ndim1 == 2 and ndim2 == 2:
        if shape1[1] != shape2[0]:
            raise NotImplementedError(
                f"Input shapes are not 'mk' and 'kn': {shape1=}, {shape2=}."
            )
        equation = "mk,kn->mn"
        return _mm_like_strategy(equation, mesh, op_schema)

    # 3. Fallback
    # For unsupported shapes (e.g., broadcasting, 4D), revert to original logic.
    return fallback_handler(op_schema)


def combine_strategies_for_matmul_backward(
    strategy_dx: OpStrategy, strategy_dw: OpStrategy, original_weight_spec: DTensorSpec
) -> OpStrategy:
    """
    Generate OpStrategy for matmul_backward by filtering and combining dx and dw strategies.

    Constraint: The weight (W) and its gradient (dW) must NOT be redistributed.
    They must match the original_weight_spec.
    """

    valid_dx_specs = list(strategy_dx.strategies)
    valid_dw_specs = list(strategy_dw.strategies)

    # Find Best Pairs according to the minimax communication cost and Construct Backward Specs
    min_cost = float("inf")
    best_spec: OpSpec | None = None

    for dx_spec in valid_dx_specs:
        for dw_spec in valid_dw_specs:
            if dx_spec is None or dx_spec.input_specs is None:
                continue
            if dw_spec is None or dw_spec.input_specs is None:
                continue
            if len(dx_spec.input_specs) < 2 or len(dw_spec.input_specs) < 2:
                continue

            costs_dx = dx_spec.redistribute_cost
            costs_dw = dw_spec.redistribute_cost

            if costs_dx is None or costs_dw is None:
                continue

            # Consistency Check:
            # The 'dy' (GradOutput) is used in both dx calculation and dw calculation.
            # We must ensure they require the same sharding for dy.
            dy_spec_in_dx = dx_spec.input_specs[0]
            dy_spec_in_dw = dw_spec.input_specs[1]
            if dy_spec_in_dx.placements != dy_spec_in_dw.placements:
                continue
            # Found a compatible pair!
            total_cost = (
                costs_dx[0][0] + costs_dw[1][0] + costs_dw[0][0] + costs_dx[1][0]
            )
            if total_cost < min_cost:
                min_cost = total_cost
                new_input_specs = [
                    dx_spec.input_specs[0],  # dy
                    dw_spec.input_specs[0],  # x
                    dx_spec.input_specs[1],  # w
                ]
                new_output_specs: tuple[DTensorSpec | None, ...] = (
                    cast(DTensorSpec, dx_spec.output_specs),
                    cast(DTensorSpec, dw_spec.output_specs),
                )
                # Combine redistribute costs for dy, x and w
                new_redistribute_cost = [
                    [costs_dx[0][0] + costs_dw[1][0]],
                    costs_dw[0],
                    costs_dx[1],
                ]

                best_spec = OpSpec(
                    output_specs=new_output_specs,
                    input_specs=new_input_specs,
                    redistribute_cost=new_redistribute_cost,
                )

    if not best_spec:
        raise RuntimeError(
            "No compatible matmul_backward strategy found when Weight is not redistributed!"
            f" The input info is: {strategy_dx=}, {strategy_dw=}, {original_weight_spec=}"
        )

    return OpStrategy([best_spec])


def _matmul_backward_3d(
    op_schema: OpSchema,
    mesh: DeviceMesh,
    shape_dy: tuple,
    shape_x: tuple,
    shape_w: tuple,
) -> tuple[OpStrategy, OpStrategy]:
    """
    Generate sharding strategies for 3D matmul backward.

    For 3D matmul backward (batched matmul), we have:
    - dy: (b, m, k) gradient of output
    - x: (b, m, n) input tensor
    - w: (n, k) weight tensor
    - dx: gradient w.r.t x, computed as dy @ w -> (b, m, n)
    - dw: gradient w.r.t w, computed as x @ dy -> (n, k)
    """

    # generate sharding strategies for: dy @ w -> dx
    if shape_dy[2] != shape_w[1]:
        raise NotImplementedError(
            f"Input shapes are not 'bmk'(dy) and 'nk'(w): {shape_dy=}, {shape_w=}."
        )
    equation_dx = "bmk,nk->bmn"
    op_schema_dx = copy.copy(op_schema)
    op_schema_dx.args_schema = (
        op_schema_dx.args_schema[0],
        op_schema_dx.args_schema[2],
    )
    strategy_dx = _mm_like_strategy(equation_dx, mesh, op_schema_dx)

    # generate sharding strategies for: x @ dy -> dw
    if shape_dy[0] != shape_x[0] or shape_dy[1] != shape_x[1]:
        raise NotImplementedError(
            f"Input shapes are not 'bmn'(x) and 'bmk'(dy): {shape_x=}, {shape_dy=}."
        )
    equation_dw = "bmn,bmk->nk"
    op_schema_dw = copy.copy(op_schema)
    op_schema_dw.args_schema = (
        op_schema_dw.args_schema[1],
        op_schema_dw.args_schema[0],
    )
    strategy_dw = _mm_like_strategy(equation_dw, mesh, op_schema_dw)

    return (strategy_dx, strategy_dw)


def _matmul_backward_2d(
    op_schema: OpSchema,
    mesh: DeviceMesh,
    shape_dy: tuple,
    shape_x: tuple,
    shape_w: tuple,
) -> tuple[OpStrategy, OpStrategy]:
    """
    Generate sharding strategies for 2D matmul backward.

    For 2D matmul backward, we have:
    - dy: (m, k) gradient of output
    - x: (m, n) input tensor
    - w: (n, k) weight tensor
    - dx: gradient w.r.t x, computed as dy @ w.T -> (m, n)
    - dw: gradient w.r.t w, computed as x.T @ dy -> (n, k)
    """
    # generate sharding strategies for: dy @ w -> dx
    if shape_dy[1] != shape_w[1]:
        raise NotImplementedError(
            f"dy[..., k] must match w[..., k] for 2D case: {shape_dy=}, {shape_w=}."
        )
    equation_dx = "mk,nk->mn"
    op_schema_dx = copy.copy(op_schema)
    op_schema_dx.args_schema = (
        op_schema_dx.args_schema[0],
        op_schema_dx.args_schema[2],
    )
    strategy_dx = _mm_like_strategy(equation_dx, mesh, op_schema_dx)

    if shape_x[1] != shape_w[0]:
        raise NotImplementedError(
            f"x[..., n] must match w[n, ...]: {shape_x=}, {shape_w=}."
        )
    equation_dw = "mn,mk->nk"
    op_schema_dw = copy.copy(op_schema)
    op_schema_dw.args_schema = (
        op_schema_dw.args_schema[1],
        op_schema_dw.args_schema[0],
    )
    strategy_dw = _mm_like_strategy(equation_dw, mesh, op_schema_dw)
    return strategy_dx, strategy_dw


def matmul_backward_sharding(
    op_schema: OpSchema, fallback_handler: Callable
) -> OpStrategy:
    """
    Custom sharding strategy for NPU matmul_backward.
    """
    mesh = op_schema.get_mesh_from_args()

    args = op_schema.args_schema
    if len(args) != 4:
        raise NotImplementedError(
            f"Only support four inputs, current input num is {len(args)}"
        )

    # pyrefly: ignore [missing-attribute]
    shape_dy = args[0].strategies[0].output_spec.shape
    # pyrefly: ignore [missing-attribute]
    shape_x = args[1].strategies[0].output_spec.shape
    # pyrefly: ignore [missing-attribute]
    shape_w = args[2].strategies[0].output_spec.shape
    ndim_dy = len(shape_dy)
    ndim_x = len(shape_x)
    ndim_w = len(shape_w)

    # generate sharding strategies for dx and dw computation, then combine them to get final strategies for backward
    if ndim_dy == 3 and ndim_x == 3 and ndim_w == 2:
        strategy_dx, strategy_dw = _matmul_backward_3d(
            op_schema, mesh, shape_dy, shape_x, shape_w
        )
        # get combined strategy for matmul backward
        return combine_strategies_for_matmul_backward(
            strategy_dx,
            strategy_dw,
            # pyrefly: ignore [missing-attribute]
            args[2].strategies[0].output_spec,
        )

    # generate sharding strategies for 2D matmul
    if ndim_dy == 2 and ndim_x == 2 and ndim_w == 2:
        strategy_dx, strategy_dw = _matmul_backward_2d(
            op_schema, mesh, shape_dy, shape_x, shape_w
        )

        # get combined strategy for matmul backward
        return combine_strategies_for_matmul_backward(
            strategy_dx,
            strategy_dw,
            # pyrefly: ignore [missing-attribute]
            args[2].strategies[0].output_spec,
        )

    # fallback to original logic.
    return fallback_handler(op_schema)


# apply op sharding patches
register_sharding_patch(aten.matmul.default, matmul_sharding)
register_sharding_patch(aten.matmul_backward.default, matmul_backward_sharding)
