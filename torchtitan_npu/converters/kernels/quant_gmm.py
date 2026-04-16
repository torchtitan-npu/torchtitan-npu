# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

import torch_npu
from einops import rearrange

from .gmm import group_size_params


class GMMFunctionMxfp8(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, weight, group_list):
        group_list = torch.cumsum(group_list, dim=0)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(
            x, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
            weight, axis=-2, dst_type=torch.float8_e4m3fn, scale_alg=1
        )

        return torch_npu.npu_grouped_matmul(
            [x_mxfp8],
            [weight_mxfp8],
            bias=None,
            scale=[weight_scale],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=x.dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list

        grad_mxfp8, grad_scale = torch_npu.npu_dynamic_mx_quant(
            grad, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
            weight, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_mxfp8],
            [rearrange(weight_mxfp8, "n h f -> n f h")],
            bias=None,
            scale=[rearrange(weight_scale, "n h f g -> n f h g")],
            per_token_scale=[grad_scale],
            group_list=group_list,
            group_type=0,
            output_dtype=grad.dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]

        x_mxfp8, x_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            x,
            group_list.to(torch.int32),
            round_mode="rint",
            dst_type=torch.float8_e4m3fn,
            blocksize=32,
        )
        grad_mxfp8, grad_scale = torch_npu.npu_grouped_dynamic_mx_quant(
            grad,
            group_list.to(torch.int32),
            round_mode="rint",
            dst_type=torch.float8_e4m3fn,
            blocksize=32,
        )
        grad_weight = torch_npu.npu_grouped_matmul(
            [x_mxfp8.t()],
            [grad_mxfp8],
            bias=None,
            scale=[grad_scale],
            per_token_scale=[rearrange(x_scale, "n h f -> h n f")],
            group_list=group_list,
            group_type=2,
            output_dtype=x.dtype,
            group_list_type=0,
            scale_dtype=torch_npu.float8_e8m0fnu,
            per_token_scale_dtype=torch_npu.float8_e8m0fnu,
            split_item=3,
        )[0]
        return grad_input, grad_weight, None


class GMMFunctionHif8(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, weight, group_list):
        group_list = torch.cumsum(group_list, dim=0)
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list
        x_quant, x_scale, w_quant, w_scale = GMMFunctionHif8.quantize(
            x, weight, torch_npu.hifloat8, torch_npu.hifloat8
        )
        gmm_kwargs = {"x_dtype": torch_npu.hifloat8, "weight_dtype": torch_npu.hifloat8}
        return torch_npu.npu_grouped_matmul(
            [x_quant],
            [w_quant],
            scale=[w_scale],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=0,
            bias=None,
            split_item=3,
            output_dtype=x.dtype,
            group_list_type=0,
            **gmm_kwargs
        )[0]

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad):
        x, weight = ctx.saved_tensors
        group_list = ctx.group_list
        weight_t = rearrange(weight, "n h f -> n f h")
        gmm_kwargs = {"x_dtype": torch_npu.hifloat8, "weight_dtype": torch_npu.hifloat8}
        grad_quant, grad_scale, w_quant, w_scale = GMMFunctionHif8.quantize(
            grad, weight_t, torch_npu.hifloat8, torch_npu.hifloat8
        )
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_quant],
            [w_quant],
            bias=None,
            scale=[w_scale],
            per_token_scale=[grad_scale],
            group_list=group_list,
            group_type=0,
            split_item=3,
            output_dtype=grad.dtype,
            group_list_type=0,
            **gmm_kwargs
        )[0]

        x_quant, x_scale, grad_quant, grad_scale = GMMFunctionHif8.quantize(
            x, grad, torch_npu.hifloat8, torch_npu.hifloat8
        )
        grad_weight = torch_npu.npu_grouped_matmul(
            [x_quant.t()],
            [grad_quant],
            scale=[grad_scale],
            per_token_scale=[x_scale],
            group_list=group_list,
            group_type=2,
            bias=None,
            split_item=3,
            output_dtype=x.dtype,
            group_list_type=0,
            **gmm_kwargs
        )[0]

        return grad_input, grad_weight, None

    @staticmethod
    def quantize(
        x: torch.Tensor,
        weight: torch.Tensor,
        x_dst_type: torch.dtype,
        w_dst_type: torch.dtype,
    ):
        g_size = group_size_params["g_size"]
        x_quant, x_scale = torch_npu.npu_dynamic_quant(
            # pyrefly: ignore [no-matching-overload]
            x.reshape(g_size, -1),
            dst_type=x_dst_type,
        )
        weight_quant, weight_scale = torch_npu.npu_dynamic_quant(
            # pyrefly: ignore [no-matching-overload]
            weight.reshape(g_size, -1),
            dst_type=w_dst_type,
        )
        return (
            x_quant.reshape(x.shape),
            x_scale,
            weight_quant.reshape(weight.shape),
            weight_scale,
        )


def npu_grouped_mxfp8_mm(x, weight, group_list):
    return GMMFunctionMxfp8.apply(x, weight, group_list)


def npu_grouped_hif8_mm(x, weight, group_list):
    return GMMFunctionHif8.apply(x, weight, group_list)
