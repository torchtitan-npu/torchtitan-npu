# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import math
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from torchtitan.components.ft import FTManager
from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import (
    LRScheduler as LRSchedulerConfig,
    Optimizer as OptimizerConfig,
)
from torchtitan.distributed import ParallelDims

logger = logging.getLogger("torchtitan")

_MUON_EXCLUDED_KEYWORDS = ("embed", "lm_head", "output")


def _should_use_muon(p: nn.Parameter, name: str) -> bool:
    """Check if parameter should be optimized by Muon.

    Rules:
    - 2D params go to Muon, except embeddings, lm_head, and output layers
    - Non-2D params go to AdamW
    """
    if p.ndim != 2:
        return False
    return not any(kw in name for kw in _MUON_EXCLUDED_KEYWORDS)


def _split_parameters_for_muon(
    model_parts: list[nn.Module],
) -> tuple[list[nn.Parameter], list[str], list[nn.Parameter], list[str]]:
    """Split parameters into Muon (2D) and AdamW (non-2D) groups.

    Returns:
        Tuple of (muon_params, muon_param_names, adamw_params, adamw_param_names)
    """
    muon_params = []
    muon_param_names = []
    adamw_params = []
    adamw_param_names = []

    for model in model_parts:
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _should_use_muon(p, name):
                muon_params.append(p)
                muon_param_names.append(name)
            else:
                adamw_params.append(p)
                adamw_param_names.append(name)

    return muon_params, muon_param_names, adamw_params, adamw_param_names


def _get_muon_lr_config(
    optimizer_config: OptimizerConfig,
    base_lr: float,
) -> tuple[float, str | None]:
    """Calculate Muon's effective learning rate and adjustment mode.

    Returns:
        Tuple of (muon_lr, muon_adjust_lr_fn)
    """
    muon_adjust_lr_fn = (
        optimizer_config.muon_adjust_lr_fn  # pyrefly: ignore[missing-attribute]
    )
    muon_lr = getattr(optimizer_config, "muon_lr", None)

    if muon_adjust_lr_fn == "original" and muon_lr is not None:
        return float(muon_lr), muon_adjust_lr_fn

    if muon_adjust_lr_fn == "match_rms_adamw" and muon_lr is not None:
        logger.warning(
            f"[Muon] muon_lr={muon_lr} is ignored when "
            f"muon_adjust_lr_fn='match_rms_adamw'. Using base lr={base_lr} instead."
        )
    return base_lr, muon_adjust_lr_fn


def _build_muon_kwargs(
    muon_lr: float,
    weight_decay: float,
    optimizer_config: OptimizerConfig,
    muon_adjust_lr_fn: str | None,
) -> dict[str, Any]:
    """Build kwargs for torch.optim.Muon constructor."""
    muon_kwargs = {
        "lr": muon_lr,
        "weight_decay": weight_decay,
        "momentum": optimizer_config.muon_momentum,  # pyrefly: ignore[missing-attribute]
        "nesterov": optimizer_config.muon_enable_nesterov,  # pyrefly: ignore[missing-attribute]
        "ns_steps": optimizer_config.muon_ns_steps,  # pyrefly: ignore[missing-attribute]
    }
    if muon_adjust_lr_fn:
        muon_kwargs[
            "adjust_lr_fn"
        ] = muon_adjust_lr_fn  # pyrefly: ignore[bad-typed-dict-key]
    return muon_kwargs


def _build_adamw_kwargs(
    lr: float,
    weight_decay: float,
    optimizer_config: OptimizerConfig,
) -> dict[str, Any]:
    """Build kwargs for torch.optim.AdamW constructor."""
    optim_implementation = optimizer_config.implementation
    if optim_implementation not in ["fused", "foreach", "for-loop"]:
        raise ValueError(
            f"Invalid implementation '{optim_implementation}'. "
            f"Must be one of: 'fused', 'foreach', 'for-loop'"
        )
    return {
        "lr": lr,
        "betas": (optimizer_config.beta1, optimizer_config.beta2),
        "eps": optimizer_config.eps,
        "weight_decay": weight_decay,
        "fused": optim_implementation == "fused",
        "foreach": optim_implementation == "foreach",
    }


def build_muon_hybrid_optimizers(
    model_parts: list[nn.Module],
    optimizer_config: OptimizerConfig,
    parallel_dims: ParallelDims,
    ft_manager: FTManager | None = None,
) -> OptimizersContainer:
    """Build Muon hybrid optimizer: Muon (for 2D params) + AdamW (for non-2D params)."""
    lr = optimizer_config.lr
    weight_decay = optimizer_config.weight_decay

    muon_lr, muon_adjust_lr_fn = _get_muon_lr_config(optimizer_config, lr)

    (
        muon_params,
        muon_param_names,
        adamw_params,
        adamw_param_names,
    ) = _split_parameters_for_muon(model_parts)

    logger.info(
        f"[MuonAdamW] Muon optimizer parameters ({len(muon_param_names)}): {muon_param_names}"
    )
    logger.info(
        f"[MuonAdamW] AdamW optimizer parameters ({len(adamw_param_names)}): {adamw_param_names}"
    )

    muon_kwargs = _build_muon_kwargs(
        muon_lr, weight_decay, optimizer_config, muon_adjust_lr_fn
    )
    adamw_kwargs = _build_adamw_kwargs(lr, weight_decay, optimizer_config)

    muon = torch.optim.Muon(
        muon_params, **muon_kwargs
    )  # pyrefly: ignore[bad-argument-type]
    adamw = torch.optim.AdamW(adamw_params, **adamw_kwargs)

    return MuonHybridOptimizersContainer(
        model_parts, [muon, adamw], muon_adjust_lr_fn=muon_adjust_lr_fn
    )


class MuonHybridOptimizersContainer(OptimizersContainer):
    """Container for Muon + AdamW hybrid optimizers.

    Key difference from upstream OptimizersContainer:
    - Upstream: model_parts[i] <-> optimizers[i] (1:1 pairing)
    - This class: each optimizer manages a subset of params from all model_parts

    When muon_adjust_lr_fn == "original":
    - Muon and AdamW use different base_lr
    - Must be used with MuonLRSchedulersContainer

    When muon_adjust_lr_fn == "match_rms_adamw":
    - Muon and AdamW use the same base_lr
    - Can use standard LRSchedulersContainer

    state_dict/load_state_dict use double loop over each optimizer and model_part.
    DCP APIs automatically filter to only process params managed by each optimizer.
    """

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizers: list[Optimizer],
        muon_adjust_lr_fn: str | None = None,
    ) -> None:
        self.model_parts = model_parts
        self.optimizers = optimizers
        self.muon_adjust_lr_fn = muon_adjust_lr_fn
        all_params = []
        for model in model_parts:
            all_params.extend(p for p in model.parameters() if p.requires_grad)
        Optimizer.__init__(self, all_params, {})

    def __iter__(self) -> Iterator[Optimizer]:
        """Return iterator over sub-optimizers for MuonLRSchedulersContainer."""
        return iter(self.optimizers)

    def __len__(self) -> int:
        """Return number of optimizers (Muon + AdamW = 2)."""
        return len(self.optimizers)

    @property
    def muon_optimizer(self) -> Optimizer:
        """Get the Muon optimizer."""
        return self.optimizers[0]

    @property
    def adamw_optimizer(self) -> Optimizer:
        """Get the AdamW optimizer."""
        return self.optimizers[1]

    def step(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Save state for all optimizers using double loop over optimizer x model_part."""
        merged = {}
        for opt in self.optimizers:
            for model in self.model_parts:
                sd = get_optimizer_state_dict(
                    model,
                    opt,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )
                merged.update(sd)
        return merged

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load state for all optimizers using double loop over optimizer x model_part."""
        for opt in self.optimizers:
            for model in self.model_parts:
                set_optimizer_state_dict(
                    model,
                    opt,
                    optim_state_dict=state_dict,
                    options=StateDictOptions(flatten_optimizer_state_dict=True),
                )


class MuonLRSchedulersContainer:
    """LR Scheduler container for Muon hybrid optimizers.

    Creates independent LambdaLR schedulers for Muon and AdamW,
    ensuring each maintains its own base_lr.

    Key difference from upstream LRSchedulersContainer:
    - Upstream: assumes all optimizers use the same base_lr
    - This class: allows Muon and AdamW to have different base_lr

    Note: state_dict only saves the first scheduler's state (last_epoch),
    consistent with upstream behavior since Muon and AdamW share the same
    lr curve, only differing in base_lr.
    """

    def __init__(
        self,
        optimizers: MuonHybridOptimizersContainer,
        lr_lambda: Callable,
    ) -> None:
        if len(optimizers) != 2:
            raise ValueError(
                f"MuonHybridOptimizersContainer must have 2 optimizers, got {len(optimizers)}"
            )

        # Create independent LambdaLR for Muon and AdamW
        self.schedulers = [
            LambdaLR(optimizers.muon_optimizer, lr_lambda),
            LambdaLR(optimizers.adamw_optimizer, lr_lambda),
        ]

        logger.info("[MuonLRSchedulersContainer] Created 2 schedulers")
        logger.info(f"  Muon scheduler base_lrs: {self.schedulers[0].base_lrs}")
        logger.info(f"  AdamW scheduler base_lrs: {self.schedulers[1].base_lrs}")
        logger.info(
            f"  Muon param_groups lr: {[pg['lr'] for pg in optimizers.muon_optimizer.param_groups]}"
        )
        logger.info(
            f"  AdamW param_groups lr: {[pg['lr'] for pg in optimizers.adamw_optimizer.param_groups]}"
        )

    def __iter__(self):
        return iter(self.schedulers)

    def __len__(self) -> int:
        return len(self.schedulers)

    def step(self) -> None:
        """Step all schedulers synchronously."""
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self) -> dict[str, Any]:
        """Save only the first scheduler's state (last_epoch is shared)."""
        return self.schedulers[0].state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load last_epoch for all schedulers without overwriting base_lrs.

        Critical: Only load last_epoch, do not overwrite base_lrs.
        base_lrs are set from each optimizer during LambdaLR construction.
        Muon and AdamW have different base_lrs that must remain independent.

        PyTorch LambdaLR.load_state_dict would overwrite base_lrs, so we
        manually set last_epoch instead of calling load_state_dict.
        """
        last_epoch = state_dict["last_epoch"]
        for scheduler in self.schedulers:
            scheduler.last_epoch = last_epoch
            scheduler._step_count = last_epoch + 1
            scheduler._last_lr = [
                scheduler.base_lrs[i] * scheduler.lr_lambdas[i](last_epoch)
                for i in range(len(scheduler.base_lrs))
            ]


def _build_lr_lambda_from_config(
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> Callable:
    """Build lr_lambda function from scheduler config.

    Extracted from build_muon_lr_schedulers to avoid code duplication.
    """
    warmup_steps = int(lr_scheduler_config.warmup_steps)

    if warmup_steps > training_steps:
        logger.warning(
            f"Warmup steps ({warmup_steps}) exceed total training steps ({training_steps}). "
            f"Adjusting warmup steps to {training_steps}."
        )
        warmup_steps = training_steps

    if lr_scheduler_config.decay_ratio is not None:
        decay_steps = round(training_steps * lr_scheduler_config.decay_ratio)
        if warmup_steps + decay_steps > training_steps:
            decay_steps = training_steps - warmup_steps
    else:
        decay_steps = training_steps - warmup_steps

    stable_steps = training_steps + 1 - warmup_steps - decay_steps
    lr_decay_type = lr_scheduler_config.decay_type
    min_lr_factor = lr_scheduler_config.min_lr_factor

    def linear_warmup_stable_decay(
        current_step: int,
        warmup_steps: int,
        stable_steps: int,
        decay_steps: int,
        lr_decay_type: str,
        min_lr_factor: float,
    ):
        warmup_stable_steps = warmup_steps + stable_steps
        if current_step < warmup_steps:
            current_step += 1
            curr_adjustment = float(current_step / warmup_steps)
        elif current_step < warmup_stable_steps:
            curr_adjustment = 1.0
        else:
            current_step += 1
            progress = float(current_step - warmup_stable_steps) / decay_steps
            if lr_decay_type == "linear":
                curr_adjustment = 1 - progress
            elif lr_decay_type == "sqrt":
                curr_adjustment = 1 - math.sqrt(progress)
            elif lr_decay_type == "cosine":
                curr_adjustment = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                raise ValueError(f"Unknown lr_decay_type: {lr_decay_type}")
            curr_adjustment = min_lr_factor + (1 - min_lr_factor) * curr_adjustment
        return curr_adjustment

    return functools.partial(
        linear_warmup_stable_decay,
        warmup_steps=warmup_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        lr_decay_type=lr_decay_type,
        min_lr_factor=min_lr_factor,
    )


def build_muon_lr_schedulers(
    optimizers: MuonHybridOptimizersContainer,
    lr_scheduler_config: LRSchedulerConfig,
    training_steps: int,
) -> MuonLRSchedulersContainer | Any:
    """Build LR scheduler for MuonHybridOptimizersContainer.

    Routes to different scheduler types based on muon_adjust_lr_fn:
    - "original": MuonLRSchedulersContainer (different base_lr for Muon and AdamW)
    - Other: Standard LRSchedulersContainer (same base_lr for both)

    Args:
        optimizers: MuonHybridOptimizersContainer instance
        lr_scheduler_config: LR scheduler configuration
        training_steps: Total training steps

    Returns:
        MuonLRSchedulersContainer or LRSchedulersContainer
    """
    lr_lambda = _build_lr_lambda_from_config(lr_scheduler_config, training_steps)

    if optimizers.muon_adjust_lr_fn == "original":
        return MuonLRSchedulersContainer(optimizers, lr_lambda)
    else:
        return LRSchedulersContainer(optimizers, lr_lambda)
