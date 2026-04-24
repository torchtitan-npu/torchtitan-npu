# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from functools import wraps

import torch
import torch._inductor.graph
import torch.nn as nn
from torch._inductor.decomposition import decompositions
from torch._inductor.lowering import lowerings

from torchtitan_npu.patches.torch._inductor.graph import graphlowering_call_function

from ..base_converter import BaseConverter
from ..convert_utils import find_functions
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)


def compile_bypass_fusion(func):
    """
    Wrapper used to clear lowerings and decompositions before torch.compile
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        lowerings.clear()
        decompositions.clear()
        return func(*args, **kwargs)

    return wrapper


@register_npu_converter("npu_bypass_triton_codegen")
class BypassTritonCodegenKernel(BaseConverter):

    SUPPORTED_MODELS = {"deepseek_v32", "llama3"}

    @classmethod
    # pyrefly: ignore [bad-override]
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> nn.Module:
        target = "apply_compile"
        pkg = "torchtitan.models"
        pkg_npu = "torchtitan_npu.models"
        matches = find_functions(target, package=pkg)
        matches.extend(find_functions(target, package=pkg_npu))
        if not matches:
            logger.info(
                "  No matched function apply_compile for this model, continue without patching"
            )
            return model

        for m in matches:
            m.replace(compile_bypass_fusion(m.func))

        # Lazy imports to avoid requiring NPU hardware at module load time
        from torchtitan_npu.patches.torch_npu._inductor.lowering import fix_npu_inductor

        torch._inductor.graph.GraphLowering.call_function = graphlowering_call_function
        fix_npu_inductor()

        # pyrefly: ignore [bad-return]
        return len(matches)
