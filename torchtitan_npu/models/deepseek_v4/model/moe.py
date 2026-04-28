# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.moe.moe import MoE, TokenChoiceTopKRouter


@dataclass
class MoEArgs:
    num_experts: int = 8
    num_shared_experts: int = 1

    # router
    score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"] = "sqrtsoftplus"
    route_norm: bool = False
    route_scale: float = 1.5
    gate_bias: bool = False
    score_before_experts: bool = False

    # token-choice with optional node limited routing
    top_k: int = 1
    num_expert_groups: int | None = 8  # must be a divisor of num_experts
    num_limited_groups: int | None = 8
    use_grouped_mm: bool = True  # grouped mm or for-loop for the experts computation
    load_balance_coeff: float | None = 1e-3

    _debug_force_load_balance: bool = False
    # if True, we force each experts get same amount of token via round-robin

    n_hash_layers: int = 3
    swiglu_limit: float = 10


def _softplus_stable(x: torch.Tensor) -> torch.Tensor:
    return torch.log1p(torch.exp(-x.abs())) + torch.relu(x)


class TokenChoiceTopKRouter(TokenChoiceTopKRouter):
    def __init__(
        self,
        dim: int,
        num_experts: int,
        top_k: int,
        layer_id: int,
        args: MoEArgs,
        score_func: Literal["softmax", "sigmoid", "sqrtsoftplus"],
        route_norm: bool,
        route_scale: float,
        vocab_size: int,
        _debug_force_load_balance: bool = False,
    ):
        super().__init__(
            dim,
            num_experts,
            args.num_expert_groups,
            args.num_limited_groups,
            top_k,
            # pyrefly: ignore [bad-argument-type]
            score_func,
            route_norm,
            route_scale,
            _debug_force_load_balance,
        )
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.num_experts = num_experts
        self.top_k = top_k
        # pyrefly: ignore [bad-assignment]
        self.score_func = score_func
        self.route_norm = route_norm
        self.route_scale = route_scale
        self._debug_force_load_balance = _debug_force_load_balance
        self.hash = layer_id < args.n_hash_layers
        self.vocab_size = vocab_size
        if self.hash:
            tid2eid = nn.Parameter(
                torch.stack(
                    [torch.randperm(self.top_k) for _ in range(self.vocab_size)]
                ),
                requires_grad=False,
            )
            self.register_buffer(
                "tid2eid",
                tid2eid,
                persistent=True,
            )

    # pyrefly: ignore [bad-override]
    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor,
        expert_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input tensor with shape ``(bs*slen, dim)``.
            expert_bias (torch.Tensor | None, optional): Optional bias tensor for experts with shape ``(num_experts,)``.
                Used for load balancing. Defaults to None.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - top_scores (torch.Tensor):
                    Routing scores for selected experts with shape ``(bs*slen, top_k)``.
                - selected_experts_indices (torch.Tensor):
                    Expert indices selected for each token with shape ``(bs*slen, top_k)``.
                - num_tokens_per_expert (torch.Tensor):
                    Number of tokens assigned to each expert with shape ``(num_experts,)``.
        """
        scores = self.gate(x)
        if self.score_func == "sigmoid":
            scores = torch.sigmoid(scores.to(torch.float32))
        elif self.score_func == "softmax":
            scores = F.softmax(scores.to(torch.float32), dim=1)
        elif self.score_func == "sqrtsoftplus":
            scores = _softplus_stable(scores.to(torch.float32)).sqrt()
        else:
            raise NotImplementedError(f"Unknown score function {self.score_func}")

        if self.hash:
            selected_experts_indices = self.tid2eid[input_ids.flatten()]
        else:
            # expert_bias is used only in non-hash layers.
            scores_for_choice = scores if expert_bias is None else scores + expert_bias
            selected_experts_indices = scores_for_choice.topk(self.top_k, dim=-1)[1]

        # top scores shape (bs*slen, top_k)
        # NOTE: The expert_bias is only used for routing. The gating value
        #       top_scores is still derived from the original scores.
        top_scores = scores.gather(dim=1, index=selected_experts_indices)
        # debug override: balanced round-robin routing
        if self._debug_force_load_balance:
            (
                selected_experts_indices,
                top_scores,
            ) = self._debug_force_load_balance_routing(scores)

        if self.route_norm:
            denominator = top_scores.sum(dim=-1, keepdim=True) + 1e-20
            top_scores = top_scores / denominator
        top_scores = top_scores * self.route_scale

        # group tokens together by expert indices from 0 to num_experts and pass that to experts forward
        num_tokens_per_expert = torch.histc(
            selected_experts_indices.view(-1),
            bins=self.num_experts,
            min=0,
            max=self.num_experts,
        )
        return top_scores, selected_experts_indices, num_tokens_per_expert

    def init_weights(self, init_std: float):
        nn.init.trunc_normal_(self.gate.weight, mean=0.0, std=init_std)
        with torch.device(self.gate.weight.device):
            if self.hash:
                self.tid2eid = torch.stack(
                    [torch.randperm(self.top_k) for _ in range(self.vocab_size)]
                )


class MoE(MoE):
    def __init__(
        self, moe_args: MoEArgs, dim: int, hidden_dim: int, layer_id, vocab_size
    ):
        # pyrefly: ignore [bad-argument-type]
        super().__init__(moe_args, dim, hidden_dim)
        # pyrefly: ignore [missing-argument]
        self.router = TokenChoiceTopKRouter(
            dim=dim,
            # pyrefly: ignore [unexpected-keyword]
            layer_id=layer_id,
            # pyrefly: ignore [unexpected-keyword]
            args=moe_args,
            num_experts=moe_args.num_experts,
            top_k=moe_args.top_k,
            # pyrefly: ignore [bad-argument-type]
            score_func=moe_args.score_func,
            route_norm=moe_args.route_norm,
            route_scale=moe_args.route_scale,
            # pyrefly: ignore [unexpected-keyword]
            vocab_size=vocab_size,
            _debug_force_load_balance=moe_args._debug_force_load_balance,
        )
        self.score_before_experts = moe_args.score_before_experts
        # pyrefly: ignore [bad-argument-type]
        self.experts.swiglu_limit = moe_args.swiglu_limit

        # Remove expert_bias buffer for hash layers. Note that init_weight of
        # class MoE will still create a non-buffer field named as `expert_bias`
        # so that `build_optimizers_with_moe_load_balancing` will not break.
        if layer_id < moe_args.n_hash_layers:
            del self.expert_bias

    # pyrefly: ignore [bad-override]
    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the DeepSeek MoE module.

        Args:
            x (torch.Tensor): Input tensor with shape ``(bs, slen, dim)``.
            input_ids (torch.Tensor): Token IDs tensor for hash-based routing.

        Returns:
            torch.Tensor: Output tensor with shape ``(bs, slen, dim)``.
        """
        bs, slen, dim = x.shape
        x_flat = x.view(-1, dim)
        input_ids_flat = input_ids.flatten()
        top_scores, selected_experts_indices, num_tokens_per_expert = self.router(
            x_flat,
            input_ids_flat,
            self.expert_bias,
        )

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert)

        (
            top_scores_experts_sorted,
            token_indices_experts_sorted,
            num_tokens_per_expert,
        ) = self.reorderer(top_scores, selected_experts_indices)

        # shape (bs*slen*top_k, dim)
        routed_input = x_flat[token_indices_experts_sorted // self.router.top_k]

        if self.score_before_experts:
            routed_input = (
                routed_input.to(torch.float32)
                * top_scores_experts_sorted.reshape(-1, 1)
            ).to(x.dtype)

        # shape (bs*slen*top_k, dim)
        routed_output = self.experts(routed_input, num_tokens_per_expert)

        # Shared expert
        shared_output = (
            self.shared_experts(x_flat) if self.shared_experts is not None else None
        )

        # Unsorted routed outputs
        routed_output_unsorted = torch.zeros(
            (bs * slen * self.router.top_k, dim),
            dtype=routed_output.dtype,
            device=routed_output.device,
        )
        routed_output_unsorted[token_indices_experts_sorted] = routed_output
        routed_output_unsorted = routed_output_unsorted.reshape(
            -1, self.router.top_k, dim
        )

        if not self.score_before_experts:
            out_experts = (
                torch.bmm(
                    top_scores.reshape(-1, 1, self.router.top_k),
                    routed_output_unsorted.float(),
                )
                .to(x.dtype)
                .squeeze(1)
            )
        else:
            out_experts = routed_output_unsorted.sum(dim=1)

        if shared_output is None:
            output = out_experts.reshape(bs, slen, dim)
        else:
            output = (shared_output + out_experts).reshape(bs, slen, dim)

        return output
