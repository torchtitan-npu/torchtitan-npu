# Adapted from
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/distributed/context_parallel.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates
# Developed by Huawei Technologies Co., Ltd. based on Meta Platforms, Inc. and affiliates TorchTitan
from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn

# pyrefly: ignore [missing-import]
import torchtitan.distributed.context_parallel as titan_cp
from torch.distributed.device_mesh import DeviceMesh


_orig_apply_cp_to_attention_module = titan_cp.apply_cp_to_attention_module


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

        from torch.distributed.tensor.experimental._attention import _ContextParallel

        from torchtitan_npu.distributed.context_parallel.dsa_cp import (
            patch_dsa_for_context_parallel,
        )

        patch_dsa_for_context_parallel(cp_mesh=cp_mesh, model_args=model_args)
        cp_plan = _ContextParallel(
            seq_dim=2, attention_type=_ContextParallel.AttentionType.SDPA
        )

        for attention_module in attention_modules:
            titan_cp.parallelize_module(  # type: ignore[attr-defined]
                module=attention_module, device_mesh=cp_mesh, parallelize_plan=cp_plan
            )
    else:
        _orig_apply_cp_to_attention_module(
            attention_modules=attention_modules,
            cp_mesh=cp_mesh,
            attention_type=attention_type,
        )
    return None


titan_cp.apply_cp_to_attention_module = apply_cp_to_attention_module
