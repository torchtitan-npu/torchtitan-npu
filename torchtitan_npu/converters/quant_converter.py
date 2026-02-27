# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import torch
import torch_npu
import torch.nn as nn

from torchtitan.config.job_config import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger
from torchtitan.components.quantization.mx import MXLinearConverter
from torchtitan.components.quantization.mx import MXGroupedMMConverter
from torchtitan.components.quantization import QuantizationConverter
from ..patches.quantization.quant_config import MXLinearConfig as TorchMXLinearConfig
from ..patches.quantization.quant_config import MoETrainingConfig as TorchMoETrainingConfig
from ..patches.quantization.quantize import linear_quantize_
from ..patches.quantization.quantize import grouped_quantize_


def is_a5():
    try:
        return "Ascend910_95" in torch_npu.npu.get_device_name()
    except Exception:
        return False


def module_filter_fn(mod: nn.Module, fqn: str, filter_fqns: list[str]) -> bool:
    """
    Filter function to determine which modules should be converted.
    For both Float8 and MXFP8, we only convert Linear modules and not matching any filtered FQNs.
    """
    if not isinstance(mod, nn.Linear):
        return False

    # If the fqn matches any filtered fqn, then we should not convert this module.
    is_filtered_fqn = any(filter_fqn in fqn for filter_fqn in filter_fqns)

    return not is_filtered_fqn


def npu_quant_linear_converter_init(self, job_config: JobConfig, parallel_dims: ParallelDims):
    QuantizationConverter._validate(job_config)
    self.enabled = False
    if not is_a5():
        raise RuntimeError("MXFP8 is only supported on Ascend910_95 or higher architecture.")

    # TP not yet supported with torch.compile
    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )
    if model_compile_enabled and job_config.parallelism.tensor_parallel_degree > 1:
        raise RuntimeError("TP not yet supported with torch.compile for mxfp8")

    mx_job_config: TorchMXLinearConfig = job_config.quantize.linear.mx
    # In NPUs, recipe_name is used to distinguish between MXFP8 and HiF8. 
    config = TorchMXLinearConfig.from_recipe_name(mx_job_config.recipe_name)
    self.filter_fqns = mx_job_config.filter_fqns
    self.config = config
    self.enabled = True
    logger.info(f"MX training active with recipe {mx_job_config.recipe_name}")


def npu_quant_linear_converter(self, model: nn.Module):
    """
    Converts the linear layers of 'model' to 'MXLinear_NPU'
    """
    if not self.enabled:
        return
    linear_quantize_(
        model,
        config=self.config,
        filter_fn=partial(module_filter_fn, filter_fqns=self.filter_fqns),
    )
    logger.info("Swapped to MXLinear_NPU layers")


def npu_quant_grouped_mm_converter_init(self, job_config: JobConfig, parallel_dims: ParallelDims):
    QuantizationConverter._validate(job_config)
    self.enabled = False
    if not is_a5():
        raise RuntimeError("MXFP8 is only supported on Ascend910_95 or higher architecture.")

    mx_job_config: TorchMoETrainingConfig = job_config.quantize.grouped_mm.mx
    # In NPUs, recipe_name is used to distinguish between MXFP8 and HiF8. 
    config = TorchMoETrainingConfig.from_recipe_name(mx_job_config.recipe_name)
    self.config = config

    self.moe_fqns = job_config.quantize.grouped_mm.mx.fqns
    self.recipe_name = job_config.quantize.grouped_mm.mx.recipe_name
    self.enabled = True
    logger.info("MXFP8 MoE training enabled")


def npu_quant_grouped_mm_converter(self, model: nn.Module):
    """
    This will use low precision grouped GEMMs with dynamic quantization using the specified MX dtype,
    rather than the default high precision grouped GEMMs, for the target MoE FQNs.
    """
    if not self.enabled:
        return
    
    def moe_module_filter_fn(mod: nn.Module, cur_fqn: str) -> bool:
        for target_fqn in self.moe_fqns:
            if target_fqn in cur_fqn:
                return True
        return False
    grouped_quantize_(model, config=self.config, filter_fn=moe_module_filter_fn)
    logger.info(
        f"Converted MoE layers matching FQNs {self.moe_fqns} "
        f"to use dynamic {self.recipe_name} quantization with scaled grouped GEMMs"
    )

MXLinearConverter.__init__ = npu_quant_linear_converter_init
MXLinearConverter.convert = npu_quant_linear_converter
MXGroupedMMConverter.__init__ = npu_quant_grouped_mm_converter_init
MXGroupedMMConverter.convert = npu_quant_grouped_mm_converter