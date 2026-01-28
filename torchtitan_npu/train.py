# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchtitan.train as titan_train


def _patch_forward_backward_step_for_dsv32():
    _original = titan_train.Trainer.forward_backward_step
    
    def wrapper_forward_backward_step(self, *args, **kwargs):
        loss = _original(self, *args, **kwargs)
        if hasattr(self.model_args, "enable_indexer_loss") and self.model_args.enable_indexer_loss:
            from torchtitan_npu.models.deepseek_v32.model.model import DSAIndexerLossLoggingHelper
            DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics()
        return loss

    titan_train.Trainer.forward_backward_step = wrapper_forward_backward_step
