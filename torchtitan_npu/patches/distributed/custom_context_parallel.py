# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/distributed/context_parallel.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
from collections.abc import Sequence

import torch
import torch.nn as nn

# pyrefly: ignore [missing-import]
import torchtitan.distributed.context_parallel as titan_cp
from torch.distributed.device_mesh import DeviceMesh


_orig_apply_cp_to_attention_module = titan_cp.apply_cp_to_attention_module


class CustomContextParallelContext:
    def __init__(
        self,
        mesh: DeviceMesh,
        *,
        buffers: list[torch.Tensor] | None = None,
        buffer_seq_dims: list[int] | None = None,
        no_restore_buffers: set[torch.Tensor] | None = None,
        load_balance: bool = False,
    ):
        self._mesh = mesh
        self._buffers = buffers or []
        self._buffer_seq_dims = buffer_seq_dims or []
        self._no_restore_buffers = no_restore_buffers or set()
        self._load_balance = load_balance
        self._ctx: contextlib.AbstractContextManager | None = None

    @torch.no_grad()
    def __enter__(self):
        from torch.distributed.tensor.experimental import context_parallel

        self._ctx = context_parallel(
            self._mesh,
            buffers=self._buffers,
            buffer_seq_dims=self._buffer_seq_dims,
            no_restore_buffers=self._no_restore_buffers,
        )
        self._ctx.__enter__()
        return self

    @torch.no_grad()
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._ctx is not None:
            return self._ctx.__exit__(exc_type, exc_val, exc_tb)
        return False


def validate_ulysses_configs(
    *,
    job_config: object | None,
    model_args: object | None,
    cp_mesh: DeviceMesh,
) -> None:
    cp_degree = cp_mesh.size()

    n_heads = getattr(model_args, "n_heads", None) if model_args is not None else None
    if n_heads is not None and n_heads % cp_degree != 0:
        raise ValueError(
            f"[ulysses] n_heads={n_heads} must be divisible by "
            f"context_parallel_degree={cp_degree}."
        )

    if job_config is not None:
        training = getattr(job_config, "training", None)
        seq_len = getattr(training, "seq_len", None) if training is not None else None
        if seq_len is not None and seq_len % cp_degree != 0:
            raise ValueError(
                f"[ulysses] seq_len={seq_len} must be divisible by "
                f"context_parallel_degree={cp_degree}."
            )

        parallelism = getattr(job_config, "parallelism", None)
        tp_degree = (
            getattr(parallelism, "tensor_parallel_degree", 1)
            if parallelism is not None
            else 1
        )
        if n_heads is not None and n_heads % (tp_degree * cp_degree) != 0:
            raise ValueError(
                f"[ulysses] n_heads={n_heads} must be divisible by "
                f"tp_degree * cp_degree = {tp_degree} * {cp_degree} = {tp_degree * cp_degree}."
            )


def validate_dsa_converters(*, job_config: object | None) -> None:
    if job_config is None:
        raise ValueError(
            '[dsa] attention_type="dsa" requires job_config (with model.converters) '
            "to be passed into apply_cp_to_attention_module."
        )
    model_cfg = getattr(job_config, "model", None)
    converters = (
        getattr(model_cfg, "converters", None) if model_cfg is not None else None
    )
    if not converters or "npu_dsa" not in converters:
        raise ValueError(
            '[dsa] attention_type="dsa" requires "npu_dsa" in job_config.model.converters. '
            f"Got converters={converters!r}."
        )


def apply_cp_to_attention_module(
    attention_modules: Sequence[nn.Module],
    cp_mesh: DeviceMesh,
    attention_type: str,
    *,
    job_config: object | None = None,
    model_args: object | None = None,
) -> None:
    if attention_type == "dsa":
        validate_dsa_converters(job_config=job_config)

        from torchtitan_npu.distributed.context_parallel.dsa_cp import (
            patch_dsa_for_context_parallel,
        )

        patch_dsa_for_context_parallel(cp_mesh=cp_mesh, model_args=model_args)
    elif attention_type == "ulysses":
        validate_ulysses_configs(
            job_config=job_config, model_args=model_args, cp_mesh=cp_mesh
        )

        from torchtitan_npu.distributed.context_parallel.ulysses_cp import (
            patch_ulysses_for_context_parallel,
        )

        patch_ulysses_for_context_parallel(cp_mesh=cp_mesh)
    else:
        _orig_apply_cp_to_attention_module(
            attention_modules=attention_modules,
            cp_mesh=cp_mesh,
            attention_type=attention_type,
        )
    return None


titan_cp.apply_cp_to_attention_module = apply_cp_to_attention_module
