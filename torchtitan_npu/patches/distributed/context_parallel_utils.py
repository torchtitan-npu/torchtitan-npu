# Adapted from
# https://github.com/pytorch/torchtitan/blob/v0.2.0/torchtitan/distributed/utils.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import functools
import threading

import torch
from torch.distributed.device_mesh import DeviceMesh
import torchtitan.distributed.utils as dist_utils
from torchtitan.train import Trainer

from torchtitan_npu.patches.tools.utils import load_class_from_string


_patch_context = threading.local()  # ceate a thread-safe context manager for monkey patching
_original_create_cp_ctx = dist_utils.create_context_parallel_ctx
_original_step_method = Trainer.forward_backward_step


@functools.wraps(_original_create_cp_ctx)
def _create_cp_ctx_wrapper(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str,
):
    # get parallel_config from the patch context
    parallel_config = getattr(_patch_context, 'current_parallel_config', None)

    # use custom cp context
    if parallel_config and getattr(parallel_config, "enable_custom_context_parallel", False):
        custom_cp_path = getattr(parallel_config, "custom_context_parallel_path", "")
        custom_cp_cls = load_class_from_string(custom_cp_path)
        if not (hasattr(custom_cp_cls, '__enter__') and hasattr(custom_cp_cls, '__exit__')):
            raise TypeError(f"Custom cp class '{custom_cp_cls}' is not a context manager with __enter__ and __exit__")
        return custom_cp_cls(
            cp_mesh,
            buffers=cp_buffers,
            buffer_seq_dims=cp_seq_dims,
            no_restore_buffers=cp_no_restore_buffers,
        )

    # use original cp context
    return _original_create_cp_ctx(cp_mesh, cp_buffers, cp_seq_dims, cp_no_restore_buffers, cp_rotate_method)


@functools.wraps(_original_step_method)
def _step_wrapper(self, *args, **kwargs):
    # before step, inject the config variable into the patch context
    _patch_context.current_parallel_config = self.job_config.parallelism
    try:
        # original step func
        return _original_step_method(self, *args, **kwargs)
    finally:
        # clear the patch context
        _patch_context.current_parallel_config = None


# patch for the cp context creation
dist_utils.create_context_parallel_ctx = _create_cp_ctx_wrapper
Trainer.forward_backward_step = _step_wrapper