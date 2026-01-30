# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from functools import wraps
import torch
import torch.nn as nn
from torch._inductor.decomposition import decompositions
from torch._inductor.lowering import lowerings

from torchtitan.config.job_config import Compile as CompileConfig
from torchtitan_npu.patches.torch._inductor.graph import graphlowering_call_function
from torchtitan_npu.patches.torch_npu._inductor.lowering import fix_npu_inductor
from torchtitan_npu.patches.torch_npu._meta_registrations import npu_fusion_attention_forward

from ..registry import (
    BaseKernel,
    KernelType,
    find_functions,
)

logger = logging.getLogger(__name__)


def compile_bypass_fusion(func):
    """
    Wrapper used to clear lowerings and decompositons before torch.compile
    """

    @wraps(func)
    def wrapper(model: nn.Module, compile_config: CompileConfig):
        lowerings.clear()
        decompositions.clear()
        return func(model, compile_config)
    
    return wrapper


class BypassTritionCodegenKernel(BaseKernel):
    kernel_type = KernelType.BypassTritionCodegen

    @classmethod
    def apply(cls, model: nn.Module, **kwargs) -> nn.Module:
        target = "apply_compile"
        pkg = "torchtitan.models"
        matches = find_functions(target, package=pkg)
        if not matches:
            logger.info(f"  No matched function apply_compile for this model, continue without patching")
            return model

        for m in matches:
            m.replace(compile_bypass_fusion(m.func))
        
        from torch_npu.op_plugin.meta._meta_registrations import npu_fusion_attention_forward as original_meta_func
        # MLA performs shape inference according to the value tensor
        original_meta_func.__code__ = npu_fusion_attention_forward.__code__
        logger.info("[Patching] npu_fusion_attention patch applied successfully.")

        torch._inductor.graph.GraphLowering.call_function = graphlowering_call_function
        fix_npu_inductor()

        logger.info(f"  Replaced: {len(matches)} apply compile functions to disable fusion triton codegen.")
        return model