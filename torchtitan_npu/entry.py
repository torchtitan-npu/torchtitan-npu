# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch

import torchtitan.distributed.activation_checkpoint as activation_checkpoint_module

from torchtitan.config.manager import ConfigManager
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer

from torchtitan_npu.patches.torchtitan.activation_checkpoint import (
    _patched_apply_full_ac,
)


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Trainer | None = None

    if config.compile.enable and config.activation_checkpoint != "none":
        logger.warning(
            "There might be performance issues with activation checkpointing and torch.compile enabled!"
        )
    else:
        activation_checkpoint_module._apply_full_ac = _patched_apply_full_ac

    if config.model.name == "deepseek_v32":
        from torchtitan_npu.train import (
            _patch_init_for_dsa_set_loss_scale,
            _patch_train_step_for_dsv32_indexer_loss,
        )

        _patch_train_step_for_dsv32_indexer_loss()
        _patch_init_for_dsa_set_loss_scale()

        from torchtitan_npu.train import _patch_for_train_npu_memory

        _patch_for_train_npu_memory()

    if config.model.name == "llama4":
        from torchtitan_npu.tools.checkpoint_patch import (
            patch_llama4_checkpoint_support,
        )

        patch_llama4_checkpoint_support()

    if config.model.name == "deepseek_v3":
        from torchtitan_npu.tools.checkpoint_patch import patch_dsv3_checkpoint_support

        patch_dsv3_checkpoint_support()

    try:
        trainer = Trainer(config)

        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed")
