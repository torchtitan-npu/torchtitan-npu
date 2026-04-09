# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/distributed/activation_checkpoint.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan/distributed/activation_checkpoint.py

Modifies activation checkpointing to check recomputation phase by using context function.
"""

import threading
from enum import Enum

import torch.nn as nn

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils._python_dispatch import TorchDispatchMode


class RecomputeState(Enum):
    FORWARD = 1
    RECOMPUTE = 2


_recompute_flag = threading.local()  # Thread-local flag for recomputation


def _indexer_loss_need_compute() -> bool:
    """
    Check if we should compute indexer loss.
    To save compute, skip it during the initial forward pass when AC is enabled.
    """
    _state = getattr(_recompute_flag, "state", None)
    return _state is None or _state == RecomputeState.RECOMPUTE


def set_recompute_state(state: RecomputeState):
    _recompute_flag.state = state


class _FwdMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **kwargs)

    def __enter__(self):
        set_recompute_state(RecomputeState.FORWARD)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        set_recompute_state(RecomputeState.RECOMPUTE)
        return super().__exit__(*args, **kwargs)


class _RecomputeMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **kwargs)

    def __enter__(self):
        set_recompute_state(RecomputeState.RECOMPUTE)
        return super().__enter__()

    def __exit__(self, *args, **kwargs):
        set_recompute_state(RecomputeState.FORWARD)
        return super().__exit__(*args, **kwargs)


def _get_context_fn():
    """Returns (forward_context, recompute_context) required by checkpoint_wrapper."""
    return (_FwdMode(), _RecomputeMode())


def _patched_apply_full_ac(module: nn.Module, ac_config) -> nn.Module:
    """Wrap module with checkpointing, injecting custom state-tracking contexts."""
    return ptd_checkpoint_wrapper(
        module,
        context_fn=_get_context_fn,
        preserve_rng_state=ac_config.preserve_rng_state,
        determinism_check=ac_config.determinism_check,
        early_stop=ac_config.early_stop,
        debug=ac_config.debug,
    )
