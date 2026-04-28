# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn

from torchtitan_npu.ops.triton import MHCPostTriton, MHCPreOnlyTriton, MHCPreTriton

from ..base_converter import BaseConverter
from ..convert_utils import replace_methods
from ..registry import register_npu_converter


def triton_hc_pre_forward(self, x, hc_fn, hc_scale, hc_base):
    r"""HcPre forward using Triton implementation.

    This function executes the "Pre-Mapping" stage of the mHC architecture. It first flattens
    the input from 4D to 3D, then applies RMSNorm normalization, computes manifold-constrained
    connection weights (`h_pre`, `h_post`, `h_res`) via linear projection and the Sinkhorn-Knopp
    algorithm, and finally aggregates the input using `h_pre` to generate the main branch output.

    Args:
        self: Module instance containing hc_mult, hc_sinkhorn_iters, hc_eps attributes
        x (torch.Tensor):
            Input tensor of shape `[B, S, N, D]`. Will be flattened to `[B, S, N*D]` internally.
        hc_fn (torch.Tensor):
            Projection weight matrix of shape `[n * n + 2 * n, n * D]`. Used to map input to the hyper-connection space.
        hc_scale (torch.Tensor):
            Branch Alpha parameters of shape `[3]`.
        hc_base (torch.Tensor):
            Branch Beta parameters of shape `[2 * n + n * n]`.

    Returns:
        y (torch.Tensor):
            Main branch output of shape `[B, S, D]`.
        h_post (torch.Tensor):
            Post-processing weight matrix of shape `[B, S, n]`.
        h_res (torch.Tensor):
            Residual weight matrix of shape `[B, S, n, n]`.
    """
    x = x.flatten(2)

    y, h_post, h_res = MHCPreTriton.apply(
        x,  # x
        hc_fn,  # weight
        hc_scale,  # branch_alpha
        hc_base,  # branch_beta
        None,  # norm_gamma
        False,  # mhc_use_gamma
        self.hc_mult,  # num_stream
        self.hc_sinkhorn_iters,  # sinkhorn_iters
        self.hc_eps,  # eps
    )
    return y, h_post, h_res


def triton_hc_post_forward(self, x, residual, post, comb):
    r"""HcPost forward using Triton implementation.

    This function executes the "Post-Mapping" stage of the mHC architecture. It flattens the
    residual from 4D to 3D, then utilizes the weights generated in the pre-stage (`h_post` and `h_res`)
    to perform a manifold-constrained weighted fusion of the current input `x` and the `residual`.

    Args:
        self: Module instance
        x (torch.Tensor):
            Current layer main input of shape `[B, S, D]`.
        residual (torch.Tensor):
            Residual input of shape `[B, S, N, D]`. Will be flattened to `[B, S, N*D]` internally.
        post (torch.Tensor):
            Post-processing weights of shape `[B, S, n]`.
        comb (torch.Tensor):
            Residual weights of shape `[B, S, n, n]`.

    Returns:
        y (torch.Tensor):
            Fused output tensor of shape `[B, S, N, D]`.
    """
    B, S, N, D = residual.shape
    residual = residual.flatten(2)

    y = MHCPostTriton.apply(
        x,  # x
        residual,  # residual
        post,  # h_post
        comb,  # h_res
    )

    y = y.view(B, S, N, D)
    return y


def triton_hc_head_forward(self, x, hc_fn, hc_scale, hc_base):
    r"""Lightweight MHC Pre-Aggregation Function (Head forward).

    Similar to `hc_pre`, but this function does not return the intermediate Sinkhorn states
    (`h_post`, `h_res`), returning only the weighted aggregated output. The input is flattened
    from 4D to 3D before processing.

    Args:
        self: Module instance containing hc_mult, hc_eps attributes
        x (torch.Tensor):
            Input tensor of shape `[B, S, N, D]`. Will be flattened to `[B, S, N*D]` internally.
        hc_fn (torch.Tensor):
            Projection weight matrix of shape `[n, n * D]`.
        hc_scale (torch.Tensor):
            Branch Alpha parameters of shape `[1]`.
        hc_base (torch.Tensor):
            Branch Beta parameters of shape `[n]`.

    Returns:
        y (torch.Tensor):
            Weighted aggregated output of shape `[B, S, D]`.
    """
    x = x.flatten(2)

    y = MHCPreOnlyTriton.apply(
        x,  # x
        hc_fn,  # weight
        hc_scale,  # branch_alpha
        hc_base,  # branch_beta
        None,  # norm_gamma
        False,  # mhc_use_gamma
        self.hc_eps,  # eps
    )
    return y


@register_npu_converter("npu_mhc_pre")
class MHCPREKernel(BaseConverter):

    TARGET_PACKAGE = "torchtitan_npu.models.deepseek_v4.model.model"
    TARGET_CLASS = "HcPre"

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> nn.Module:

        replacement_counts = 0
        # Replace HcPre.forward
        replacement_counts += replace_methods(
            class_name=cls.TARGET_CLASS,
            method_name="forward",
            new_method=triton_hc_pre_forward,
            package=cls.TARGET_PACKAGE,
        )

        # pyrefly: ignore [bad-return]
        return replacement_counts


@register_npu_converter("npu_mhc_post")
class MHCPOSTKernel(BaseConverter):

    TARGET_PACKAGE = "torchtitan_npu.models.deepseek_v4.model.model"
    TARGET_CLASS = ["HcPost", "HcHead"]

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> nn.Module:

        replacement_counts = 0
        # Replace HcPost.forward
        replacement_counts += replace_methods(
            class_name=cls.TARGET_CLASS[0],
            method_name="forward",
            new_method=triton_hc_post_forward,
            package=cls.TARGET_PACKAGE,
        )
        # Replace HcHead.forward
        replacement_counts += replace_methods(
            class_name=cls.TARGET_CLASS[1],
            method_name="forward",
            new_method=triton_hc_head_forward,
            package=cls.TARGET_PACKAGE,
        )

        # pyrefly: ignore [bad-return]
        return replacement_counts
