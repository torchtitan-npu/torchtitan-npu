# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass

from torchtitan.config import JobConfig
from torchtitan.models.deepseek_v3.model.args import DeepSeekV3ModelArgs


@dataclass
class DeepSeekV3ModelArgs(DeepSeekV3ModelArgs):
    def update_from_config(self, job_config: JobConfig, **kwargs) -> None:
        original_use_grouped_mm = self.moe_args.use_grouped_mm

        super().update_from_config(job_config, **kwargs)
        self.moe_args.use_grouped_mm = original_use_grouped_mm
