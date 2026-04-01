# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Patch for torchtitan lr_scheduler to support MuonHybridOptimizersContainer.

When muon_adjust_lr_fn == "original", Muon and AdamW use different base_lr,
requiring MuonLRSchedulersContainer instead of standard LRSchedulersContainer.
"""

import logging

from torchtitan.components.lr_scheduler import LRSchedulersContainer

from torchtitan.components.optimizer import OptimizersContainer
from torchtitan.config import LRScheduler as LRSchedulerConfig

from torchtitan_npu.patches.optimizer.muon_optimizer import (
    build_muon_lr_schedulers,
    MuonHybridOptimizersContainer,
)

logger = logging.getLogger("torchtitan")


def _patch_build_lr_schedulers():
    """Patch build_lr_schedulers to support Muon hybrid optimizers."""
    try:
        import torchtitan.components.lr_scheduler as lr_scheduler_module
    except ImportError:
        logger.warning(
            "[lr_scheduler patch] torchtitan.components.lr_scheduler not found, skipping patch"
        )
        return

    _original_build_lr_schedulers = lr_scheduler_module.build_lr_schedulers

    def patched_build_lr_schedulers(
        optimizers: OptimizersContainer,
        lr_scheduler_config: LRSchedulerConfig,
        training_steps: int,
    ) -> LRSchedulersContainer:
        """
        Patched build_lr_schedulers that supports MuonHybridOptimizersContainer.

        When optimizers is a MuonHybridOptimizersContainer with muon_adjust_lr_fn == "original",
        use MuonLRSchedulersContainer to ensure Muon and AdamW maintain independent base_lr.
        Otherwise, fall back to standard LRSchedulersContainer.
        """
        if isinstance(optimizers, MuonHybridOptimizersContainer):
            logger.info(
                f"[lr_scheduler patch] Detected MuonHybridOptimizersContainer, "
                f"muon_adjust_lr_fn={optimizers.muon_adjust_lr_fn}"
            )
            return build_muon_lr_schedulers(  # pyrefly: ignore[bad-return]
                optimizers, lr_scheduler_config, training_steps
            )

        # Fallback to original for other optimizer types
        return _original_build_lr_schedulers(
            optimizers, lr_scheduler_config, training_steps
        )

    # Apply patch
    lr_scheduler_module.build_lr_schedulers = patched_build_lr_schedulers
    logger.info("[lr_scheduler patch] Successfully patched build_lr_schedulers")


# Auto-apply patch when module is imported
_patch_build_lr_schedulers()
