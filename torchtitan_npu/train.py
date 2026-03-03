# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import wraps

import torchtitan.train as titan_train
from torchtitan.config import JobConfig


def _patch_forward_backward_step_for_dsv32():
    _original = titan_train.Trainer.forward_backward_step
    
    def wrapper_forward_backward_step(self, *args, **kwargs):
        loss = _original(self, *args, **kwargs)
        if hasattr(self.model_args, "enable_indexer_loss") and self.model_args.enable_indexer_loss:
            from torchtitan_npu.models.deepseek_v32.model.model import DSAIndexerLossLoggingHelper
            DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics()
        return loss

    titan_train.Trainer.forward_backward_step = wrapper_forward_backward_step


def _patch_init_for_dsa_set_loss_scale():
    _original = titan_train.Trainer.__init__

    @wraps(_original)
    def wrapper_init(self, job_config: JobConfig):
        _original(self, job_config)

        if not self.parallel_dims.pp_enabled:
            return
        
        from torchtitan_npu.models.deepseek_v32.model.model import DSAIndexerLossAutoScaler
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
            
        DSAIndexerLossAutoScaler.set_loss_scale(1.0 / num_microbatches)

    titan_train.Trainer.__init__ = wrapper_init

