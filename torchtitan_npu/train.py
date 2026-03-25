# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps

import torch
import torchtitan.train as titan_train
from torchtitan.config import JobConfig
from torchtitan.tools.logging import init_logger, logger

init_logger()


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
                ga_steps = self.gradient_accumulation_steps
                if self.parallel_dims.pp_enabled:
                    local_bs = self.job_config.training.local_batch_size
                    micro_bs = (
                        self.job_config.parallelism.pipeline_parallel_microbatch_size
                    )
                    pp_microbatches = local_bs // micro_bs
                    total_acc_steps = ga_steps * pp_microbatches
                else:
                    total_acc_steps = ga_steps
                # Pass the total gradient accumulation steps from the Trainer instance
                DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics(
                    total_acc_steps=total_acc_steps
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

        if not self.parallel_dims.pp_enabled:
            return

        from torchtitan_npu.models.deepseek_v32.model.model import (
            DSAIndexerLossAutoScaler,
        )

        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            batch_degree = batch_mesh.size()
        else:
            batch_degree = 1

        global_batch_size = job_config.training.global_batch_size
        if global_batch_size < 0:
            num_microbatches = job_config.training.local_batch_size
        else:
            num_microbatches = global_batch_size // batch_degree

        DSAIndexerLossAutoScaler.set_loss_scale(torch.tensor(1.0 / num_microbatches))

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
