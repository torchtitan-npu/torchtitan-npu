# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch_npu
from torch import nn, Tensor

from torch.distributed._functional_collectives import (
    all_to_all_single,
    all_to_all_single_autograd,
)
from torch.distributed.tensor import DeviceMesh

from ..base_converter import BaseConverter
from ..convert_utils import replace_methods
from ..registry import register_npu_converter

from .permutation import NPUMoeTokenUnpermute

logger = logging.getLogger(__name__)


def _npu_moe_token_dispatch(
    self, mod: nn.Module, inputs: tuple, device_mesh: DeviceMesh
) -> tuple[Tensor, Tensor]:
    # annotate module input placements/sharding with input_layouts
    routed_input, num_tokens_per_expert = inputs
    ep_degree = device_mesh.shape[0]
    num_local_experts = num_tokens_per_expert.shape[0] // ep_degree

    # generate the input splits and output splits for all-to-all
    with torch.no_grad():
        num_tokens_per_expert_group = all_to_all_single(
            num_tokens_per_expert,
            None,
            None,
            group=device_mesh.get_group(),
        )
        # Need to wait explicitly because it is used by a triton kernel later
        # which doesn't realize that AsyncCollectiveTensor needs unwrapping
        num_tokens_per_expert_group = torch.ops._c10d_functional.wait_tensor(
            num_tokens_per_expert_group
        )
        input_splits = (
            num_tokens_per_expert.view(ep_degree, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=True)
        )
        # NOTE: this would incur a device-to-host sync
        output_splits = (
            num_tokens_per_expert_group.view(ep_degree, -1)
            .sum(dim=1)
            .to(torch.device("cpu"), non_blocking=False)
        )
        self.input_splits = input_splits.tolist()
        self.output_splits = output_splits.tolist()

    # perform all-to-all
    routed_input = all_to_all_single_autograd(
        routed_input,
        self.output_splits,
        self.input_splits,
        device_mesh.get_group(),
    )

    # NOTE: After this all-to-all, the routed input is put on proper EP rank.
    # However, the num_tokens_per_expert_group is not of the final target format
    # [#tokens for local expert 0, #tokens for local expert 1, ...]
    # Rather, it is of the format
    # [#tokens for local expert 0 from EP rank 0, #tokens for local expert 1 from EP rank 0, ...,
    #  #tokens for local expert 0 from EP rank 1, #tokens for local expert 1 from EP rank 1, ...]
    # We need to perform another shuffle to get the correct layout, via the _permute function
    # below, which also does padding to make sure the number of tokens each expert gets locally
    # is a multiple of TOKEN_GROUP_ALIGN_SIZE_M.
    # Note that this will create side effects when wrapping the for-loop implementation
    # of GroupedExperts, as it does not need padding.
    indices = None
    with torch.no_grad():
        indices = (
            torch.arange(
                num_local_experts,
                dtype=torch.int64,
                device=routed_input.device,
            )
            .repeat(ep_degree)
            .repeat_interleave(
                num_tokens_per_expert_group.view(-1),
                output_size=sum(self.output_splits),
            )
        )

    routed_input, self.permuted_indices = torch_npu.npu_moe_token_permute(
        routed_input, indices
    )

    num_tokens_per_expert_group = num_tokens_per_expert_group.view(ep_degree, -1).sum(0)

    return routed_input, num_tokens_per_expert_group


def _npu_moe_token_combine(
    self, mod: nn.Module, routed_output: Tensor, device_mesh: DeviceMesh
) -> Tensor:
    # Using NPUMoeTokenUnpermute.apply and npu_moe_token_unpermute is equivalent here,
    # and avoid storing tensor routed_output during backpropagation.
    routed_output = NPUMoeTokenUnpermute.apply(
        routed_output, self.permuted_indices, routed_output.shape
    )
    routed_output = all_to_all_single_autograd(
        routed_output,
        self.input_splits,
        self.output_splits,
        device_mesh.get_group(),
    )
    return routed_output


@register_npu_converter("npu_expert_parallel")
class ExpertParallelConverter(BaseConverter):

    DIST_PACKAGE = "torchtitan.distributed"

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> nn.Module:
        dist_pkg = cls.DIST_PACKAGE

        # 1. Replacing ExpertParallel method _token_dispatch
        counts = replace_methods(
            class_name="ExpertParallel",
            method_name="_token_dispatch",
            new_method=_npu_moe_token_dispatch,
            package=dist_pkg,
        )
        # 2. Replacing ExpertParallel method _token_combine
        counts += replace_methods(
            class_name="ExpertParallel",
            method_name="_token_combine",
            new_method=_npu_moe_token_combine,
            package=dist_pkg,
        )

        # pyrefly: ignore [bad-return]
        return counts
