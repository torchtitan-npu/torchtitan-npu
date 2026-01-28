# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch_npu
from torch import nn
from torch.distributed.tensor import DTensor

from torchtitan.models.moe.moe import indices_padding_wrapper
from ..registry import BaseKernel, KernelType, replace_methods, replace_functions

logger = logging.getLogger(__name__)


class GMMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, group_list):
        ctx.save_for_backward(x, weight)
        ctx.group_list = group_list

        fwd_output = torch_npu.npu_grouped_matmul(
            [x],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        return fwd_output

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        group_list = ctx.group_list

        weight = torch.transpose(weight, 1, 2)
        grad_input = torch_npu.npu_grouped_matmul(
            [grad_output],
            [weight],
            bias=None,
            group_list=group_list,
            split_item=2,
            group_type=0,
            group_list_type=1,
        )[0]
        grad_weight = torch_npu.npu_grouped_matmul(
            [input_tensor.T],
            [grad_output],
            bias=None,
            group_list=group_list,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]
        return grad_input, grad_weight, None


def npu_grouped_mm(x, weight, group_list):
    return GMMFunction.apply(x, weight, group_list)


def _run_experts_grouped_mm(
    w13: torch.Tensor,
    w2: torch.Tensor,
    _w3: torch.Tensor,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor | None,
) -> torch.Tensor:
    offsets = num_tokens_per_expert.to(torch.int64)

    h = npu_grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offsets)
    h = torch_npu.npu_swiglu(h, dim=-1)
    out = npu_grouped_mm(h, w2.bfloat16().transpose(-2, -1), offsets).type_as(x)

    return out


def npu_grouped_experts_forward(
    self,
    x: torch.Tensor,
    num_tokens_per_expert: torch.Tensor,
) -> torch.Tensor:
    if isinstance(self.w2, DTensor):
        # Convert parameters from DTensors to plain Tensors, to work with
        # dynamic-shape inputs in EP which cannot be easily expressed as DTensors.
        w2 = self.w2.to_local()
        w13 = self.w13.to_local() if self.w13 is not None else None
    else:
        w2 = self.w2
        w13 = self.w13

    # NOTE: If EP is not used, we need to pad the indices
    #       to prepare for grouped_mm;
    #       otherwise, EP will handle the padding.
    if (
        not isinstance(self.w2, DTensor)
        or "ep" not in self.w2.device_mesh.mesh_dim_names
    ):
        run_experts_fn = indices_padding_wrapper(_run_experts_grouped_mm)
    else:
        run_experts_fn = _run_experts_grouped_mm
    return run_experts_fn(w13, w2, None, x, num_tokens_per_expert)


def npu_grouped_experts_init_weights(self, init_std: float):
    for w in [self.w2, self.w13]:
        if w is not None:
            nn.init.normal_(w, mean=0.0, std=init_std)


class GMMKernel(BaseKernel):
    kernel_type = KernelType.GMM

    TARGET_PACKAGE = "torchtitan.models.moe"
    TARGET_CLASS = "GroupedExperts"

    @classmethod
    def apply(cls, model: nn.Module, **kwargs) -> nn.Module:

        replacement_counts = 0

        # 1. Replacing GroupedExperts methods
        replacement_counts += replace_methods(
            class_name=cls.TARGET_CLASS,
            method_name="forward",
            new_method=npu_grouped_experts_forward,
            package=cls.TARGET_PACKAGE
        )

        replacement_counts += replace_methods(
            class_name=cls.TARGET_CLASS,
            method_name="init_weights",
            new_method=npu_grouped_experts_init_weights,
            package=cls.TARGET_PACKAGE
        )

        # 2. Replacing module function _run_experts_grouped_mm
        func_replacements = replace_functions(
            func_name="_run_experts_grouped_mm",
            new_func=_run_experts_grouped_mm,
            package=cls.TARGET_PACKAGE
        )
        replacement_counts += func_replacements

        # Initialize w13
        cls._change_existing_instances(model)

        logger.info(f"  [GMM] Replaced {replacement_counts} GMM methods/functions.")

        return model

    @classmethod
    def _change_existing_instances(cls, model: nn.Module):
        """Traverse the model and convert w1+w3 of the existing GroupedExperts into w13."""
        for name, module in model.named_modules():
            class_name = type(module).__name__
            if 'GroupedExperts' not in class_name and cls.TARGET_CLASS not in class_name:
                continue
            w1 = getattr(module, 'w1', None)
            w3 = getattr(module, 'w3', None)

            if w1 is not None and w3 is not None:
                try:
                    cls._create_w13_from_w1_w3(module, name)
                except Exception as e:
                    logger.warning(f"Failed to convert {name}: {e}")
            else:
                logger.warning(f"  {name}: Missing w1/w3, skipping")
        return

    @classmethod
    def _create_w13_from_w1_w3(cls, module: nn.Module, module_name: str):
        """Create parameter w13 from w1"""
        w1 = module.w1

        num_experts = w1.shape[0]
        hidden_dim = w1.shape[1]
        dim = w1.shape[2]

        w13_data = torch.empty(num_experts, hidden_dim * 2, dim, dtype=w1.dtype, device=w1.device)
        module.register_parameter('w13', nn.Parameter(w13_data))
        module.use_grouped_mm = True

        module.w1 = None
        module.w3 = None

        logger.info(f"  {module_name}: Created w13 [{w13_data.shape}]")