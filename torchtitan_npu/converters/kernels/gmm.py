# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Any

import torch

import torch_npu
from torch import nn
from torch.distributed.tensor import DTensor
from torchtitan.models.moe.moe import indices_padding_wrapper

from ..base_converter import BaseConverter
from ..convert_utils import replace_functions, replace_methods
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)

# Calculate the number of experts and EP degree, which are used as parameters
# when invoking operators during Hifloat8 low-precision training.
group_size_params = {
    "num_experts": None,
    "expert_model_parallel_size": None,
    "g_size": None,
}


class GMMFunction(torch.autograd.Function):
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, x, weight, group_list) -> Any:
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
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output) -> Any:
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
    num_tokens_per_expert: torch.Tensor,
    swiglu_limit: float | None,
) -> torch.Tensor:
    # pyrefly: ignore [missing-attribute]
    offsets = num_tokens_per_expert.to(torch.int64)

    h = npu_grouped_mm(x.bfloat16(), w13.bfloat16().transpose(-2, -1), offsets)
    if swiglu_limit is not None:
        gate, up = h.chunk(2, -1)
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
        h = torch.cat([gate, up], dim=-1)
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
        # pyrefly: ignore [not-iterable]
        or "ep" not in self.w2.device_mesh.mesh_dim_names
    ):
        group_size_params["expert_model_parallel_size"] = 1
    else:
        # pyrefly: ignore [missing-attribute]
        ep_dim_index = self.w2.device_mesh.mesh_dim_names.index("ep")
        group_size_params["expert_model_parallel_size"] = self.w2.device_mesh.shape[
            ep_dim_index
        ]
    run_experts_fn = _run_experts_grouped_mm

    if group_size_params["g_size"] is None:
        group_size_params["num_experts"] = self.num_experts
        group_size_params["g_size"] = (
            # pyrefly: ignore [unsupported-operation]
            group_size_params["num_experts"]
            // group_size_params["expert_model_parallel_size"]
        )

    # XXX: Refactor this, only DSv4 inject this attribute to its experts.
    swiglu_limit = getattr(self, "swiglu_limit", None)

    # pyrefly: ignore [bad-argument-type]
    return run_experts_fn(w13, w2, None, x, num_tokens_per_expert, swiglu_limit)


def npu_grouped_experts_init_weights(self, init_std: float):
    for w in [self.w2, self.w13]:
        if w is not None:
            nn.init.normal_(w, mean=0.0, std=init_std)


@register_npu_converter("npu_gmm")
class GMMKernel(BaseConverter):

    TARGET_PACKAGE = "torchtitan.models.moe"
    TARGET_CLASS = "GroupedExperts"

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:

        replacement_counts = 0

        # 1. Replacing GroupedExperts methods
        replacement_counts += replace_methods(
            class_name=cls.TARGET_CLASS,
            method_name="forward",
            new_method=npu_grouped_experts_forward,
            package=cls.TARGET_PACKAGE,
        )

        replacement_counts += replace_methods(
            class_name=cls.TARGET_CLASS,
            method_name="init_weights",
            new_method=npu_grouped_experts_init_weights,
            package=cls.TARGET_PACKAGE,
        )

        # 2. Replacing module function _run_experts_grouped_mm
        func_replacements = replace_functions(
            func_name="_run_experts_grouped_mm",
            new_func=_run_experts_grouped_mm,
            package=cls.TARGET_PACKAGE,
        )
        replacement_counts += func_replacements

        # Initialize w13
        cls._change_existing_instances(model)

        # pyrefly: ignore [bad-return]
        return replacement_counts

    @classmethod
    def _change_existing_instances(cls, model: nn.Module):
        """Traverse the model and convert w1+w3 of the existing GroupedExperts into w13."""
        for name, module in model.named_modules():
            class_name = type(module).__name__
            if (
                "GroupedExperts" not in class_name
                and cls.TARGET_CLASS not in class_name
            ):
                continue
            w1 = getattr(module, "w1", None)
            w3 = getattr(module, "w3", None)

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

        # pyrefly: ignore [bad-index]
        num_experts = w1.shape[0]
        # pyrefly: ignore [bad-index]
        hidden_dim = w1.shape[1]
        # pyrefly: ignore [bad-index]
        dim = w1.shape[2]

        # pyrefly: ignore [no-matching-overload]
        w13_data = torch.empty(
            num_experts, hidden_dim * 2, dim, dtype=w1.dtype, device=w1.device
        )
        module.register_parameter("w13", nn.Parameter(w13_data))
        # pyrefly: ignore [bad-argument-type]
        module.use_grouped_mm = True

        # pyrefly: ignore [bad-argument-type]
        module.w1 = None
        # pyrefly: ignore [bad-argument-type]
        module.w3 = None

        logger.info(f"  {module_name}: Created w13 [{w13_data.shape}]")
