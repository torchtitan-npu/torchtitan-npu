# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

import torch_npu

from torchtitan_npu.patches.quantization.quant_config import (
    MXLinearConfig,
    MXLinearRecipeName,
)


def view_as_n_dim(input_tensor, dim=2):
    if dim < 2:
        raise AssertionError("dim should be greater than or equal to 2")
    if len(input_tensor.shape) != dim:
        return input_tensor.view(-1, *input_tensor.shape[-dim + 1 :])
    return input_tensor


class MXfp8MM(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, weight):
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(
            view_as_n_dim(x), axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
            weight, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        output = torch_npu.npu_quant_matmul(
            x_mxfp8,
            weight_mxfp8.t(),
            weight_scale.transpose(0, 1),
            pertoken_scale=x_scale,
            output_dtype=x.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        if len(x.shape) != 2:
            output = output.reshape(*x.shape[:-1], *output.shape[1:])
        if weight.requires_grad:
            output.requires_grad = True
        ctx.save_for_backward(x, weight)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grads):
        x, weight = ctx.saved_tensors
        grads_mxfp8, grads_scale = torch_npu.npu_dynamic_mx_quant(
            view_as_n_dim(grads), axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        weight_mxfp8, weight_scale = torch_npu.npu_dynamic_mx_quant(
            weight, axis=-2, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        dx = torch_npu.npu_quant_matmul(
            grads_mxfp8,
            weight_mxfp8,
            weight_scale,
            pertoken_scale=grads_scale,
            output_dtype=x.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        if len(grads.shape) != 2:
            dx = dx.reshape(*grads.shape[:-1], *dx.shape[1:])

        grads_mxfp8, grads_scale = torch_npu.npu_dynamic_mx_quant(
            view_as_n_dim(grads), axis=-2, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        x_mxfp8, x_scale = torch_npu.npu_dynamic_mx_quant(
            view_as_n_dim(x), axis=-2, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        dw = torch_npu.npu_quant_matmul(
            grads_mxfp8.t(),
            x_mxfp8,
            x_scale,
            pertoken_scale=grads_scale.transpose(0, 1),
            output_dtype=x.dtype,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            group_sizes=[1, 1, 32],
        )
        return dx, dw, None, None


class Hif8MM(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, weight):
        mm_kwargs = {"x1_dtype": torch_npu.hifloat8, "x2_dtype": torch_npu.hifloat8}
        x_quant, x_scale = torch_npu.npu_dynamic_quant(
            x, dst_type=torch_npu.hifloat8, quant_mode="pertensor"
        )
        w_quant, w_scale = torch_npu.npu_dynamic_quant(
            weight, dst_type=torch_npu.hifloat8, quant_mode="pertensor"
        )

        output = torch_npu.npu_quant_matmul(
            x_quant,
            w_quant.t(),
            w_scale,
            pertoken_scale=x_scale,
            output_dtype=x.dtype,
            **mm_kwargs,
        )

        if weight.requires_grad:
            output.requires_grad = True
        ctx.save_for_backward(x, weight)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grads):
        mm_kwargs = {"x1_dtype": torch_npu.hifloat8, "x2_dtype": torch_npu.hifloat8}
        x, weight = ctx.saved_tensors
        w_quant, w_scale = torch_npu.npu_dynamic_quant(
            weight, dst_type=torch_npu.hifloat8, quant_mode="pertensor"
        )
        grads_quant, grads_scale = torch_npu.npu_dynamic_quant(
            grads, dst_type=torch_npu.hifloat8, quant_mode="pertensor"
        )
        dx = torch_npu.npu_quant_matmul(
            grads_quant,
            w_quant,
            w_scale,
            pertoken_scale=grads_scale,
            output_dtype=x.dtype,
            **mm_kwargs,
        )

        x_quant, x_scale = torch_npu.npu_dynamic_quant(
            x, dst_type=torch_npu.hifloat8, quant_mode="pertensor"
        )
        grads_quant, grads_scale = torch_npu.npu_dynamic_quant(
            view_as_n_dim(grads).t(),
            dst_type=torch_npu.hifloat8,
            quant_mode="pertensor",
        )

        dw = torch_npu.npu_quant_matmul(
            grads_quant,
            view_as_n_dim(x_quant),
            x_scale,
            pertoken_scale=grads_scale,
            output_dtype=x.dtype,
            **mm_kwargs,
        )

        return dx, dw, None, None


class MXLinear(torch.nn.Linear):
    """
    Input, weight and grad_output can have each their own MX element dtype.
    """

    config: MXLinearConfig

    @classmethod
    @torch.no_grad()
    def from_float(
        cls,
        mod,
        config: MXLinearConfig | None = MXLinearConfig(),
    ):
        if not isinstance(mod, torch.nn.Linear):
            raise RuntimeError(
                f"Unsupported module type: {type(mod)}. Expected torch.nn.Linear"
            )
        mod.__class__ = MXLinear
        mod.config = config  # pyrefly: ignore [bad-argument-type]
        return mod

    def forward(self, input):
        w = self.weight
        config = self.config
        if config.recipe_name == MXLinearRecipeName.FLOAT8_MXFP8:
            y = MXfp8MM.apply(input, w)
        elif config.recipe_name == MXLinearRecipeName.FLOAT8_HIF8:
            y = Hif8MM.apply(input, w)
        if self.bias is not None:
            # pyrefly: ignore [unbound-name]
            y = y + self.bias
        # pyrefly: ignore [unbound-name]
        return y
