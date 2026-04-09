# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/train.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
from functools import wraps

import torch
import torchtitan.train as titan_train
from torchtitan.config import JobConfig
from torchtitan.tools.logging import init_logger, logger

from torchtitan_npu.converters.convert_utils import find_functions

init_logger()


def get_grad_accumulation_steps(trainer) -> int:
    ga_steps = trainer.gradient_accumulation_steps
    if trainer.parallel_dims.pp_enabled:
        local_bs = trainer.job_config.training.local_batch_size
        micro_bs = trainer.job_config.parallelism.pipeline_parallel_microbatch_size
        return ga_steps * (local_bs // micro_bs)
    return ga_steps


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:

    ndim = x.ndim
    assert ndim > 1
    seqlen = x.shape[1]
    if positions is None:
        freqs_cis = freqs_cis[0:seqlen]
        assert freqs_cis.shape == (seqlen, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    elif positions.size(0) == 1:
        assert positions.shape == (1, seqlen)
        freqs_cis_real = torch.view_as_real(freqs_cis)
        freqs_cis_real = freqs_cis_real[positions.squeeze(0)]
        freqs_cis = torch.view_as_complex(freqs_cis_real)
        assert freqs_cis.shape == (seqlen, x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)
    else:
        assert positions.shape == (x.shape[0], seqlen)
        freqs_cis_real = torch.view_as_real(freqs_cis)
        freqs_cis_real = freqs_cis_real[positions]
        freqs_cis = torch.view_as_complex(freqs_cis_real)
        shape = [x.shape[0], seqlen, 1, freqs_cis.shape[-1]]
        return freqs_cis.view(*shape)


def _patch_torchtitan_model_reshape_for_broadcast():
    torchtitan_reshape_for_broadcast_modules = [
        "torchtitan.models.deepseek_v3.model.model",
        "torchtitan.models.llama3.model.model",
        "torchtitan.models.llama4.model.model",
    ]
    total = 0
    for module_path in torchtitan_reshape_for_broadcast_modules:
        importlib.import_module(module_path)
        matches = find_functions("reshape_for_broadcast", package=module_path)
        for match in matches:
            match.replace(reshape_for_broadcast)
        total += len(matches)

    logger.info(
        f"[RoPE Broadcast Patch] Applied {total} reshape_for_broadcast patch(es)"
    )


def _patch_train_step_for_dsv32_indexer_loss():
    _original_train_step = titan_train.Trainer.train_step

    def wrapper_train_step(self, *args, **kwargs):
        # Execute the original train_step (which includes gradient accumulation loops)
        result = _original_train_step(self, *args, **kwargs)

        # Handle indexer loss tracking
        if (
            hasattr(self.model_args, "enable_indexer_loss")
            and self.model_args.enable_indexer_loss
        ):
            # Import dynamically to avoid circular dependencies
            from torchtitan_npu.models.deepseek_v32.model.model import (
                DSAIndexerLossLoggingHelper,
            )

            # Align logging frequency with the core metrics processor
            if self.metrics_processor.should_log(self.step):
                DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics(
                    total_acc_steps=get_grad_accumulation_steps(self)
                )
            else:
                # Crucial: Clear tracker silently if this step is not being logged
                # to prevent runaway accumulation of losses across steps.
                DSAIndexerLossLoggingHelper.clean_loss_in_tracker()

        return result

    # Apply the monkey patch
    titan_train.Trainer.train_step = wrapper_train_step


def _patch_init_for_dsa_set_loss_scale():
    _original = titan_train.Trainer.__init__

    @wraps(_original)
    def wrapper_init(self, job_config: JobConfig):
        _original(self, job_config)

        from torchtitan_npu.models.deepseek_v32.model.model import (
            DSAIndexerLossAutoScaler,
        )

        loss_degree = 1
        if getattr(self.parallel_dims, "dp_cp_enabled", False):
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
            if loss_mesh is not None:
                loss_degree = loss_mesh.size()

        scale = 1.0 / float(get_grad_accumulation_steps(self) * loss_degree)
        DSAIndexerLossAutoScaler.set_loss_scale(
            torch.tensor(scale, device=self.device, dtype=torch.float32)
        )

    titan_train.Trainer.__init__ = wrapper_init


def _patch_for_train_npu_memory():
    _original = titan_train.Trainer.train

    def wrapper_train(self):
        torch.npu.empty_cache()  # pyrefly: ignore[missing-attribute]
        memory_ratio = self.job_config.training.torch_npu_memory_ratio
        if not (0.0 < memory_ratio <= 1.0):
            logger.warning(
                f"torch_npu_memory_ratio {memory_ratio} is invalid "
                "(must be in (0.0, 1.0]), falling back to default value 1.0"
            )
            memory_ratio = 1.0
        torch.npu.set_per_process_memory_fraction(  # pyrefly: ignore[missing-attribute]
            memory_ratio
        )
        logger.info(
            f"[NPU Memory Config] Set process memory usage upper limit to {memory_ratio}"
        )
        return _original(self)

    titan_train.Trainer.train = wrapper_train
