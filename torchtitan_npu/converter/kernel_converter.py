# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

from torchtitan.protocols.model_converter import register_model_converter

from .base_converter import BaseNPUConverter
from .registry import KernelType

logger = logging.getLogger(__name__)


def _ensure_kernels_loaded() -> None:
    """Ensure all kernel modules are loaded"""
    kernel_modules = [
        "rms_norm",
        "rope",
        "permute",
        "gmm",
        "dsa",
        "fusion_attention",
    ]

    features_modules = [
        "bypass_triton_codegen"
    ]

    import importlib
    for name in kernel_modules:
        try:
            importlib.import_module(f".kernels.{name}", package=__package__)
        except ImportError as e:
            logger.error(f"Kernel module {name} not available: {e}")

    for name in features_modules:
        try:
            importlib.import_module(f".features.{name}", package=__package__)
        except ImportError as e:
            logger.error(f"Feature module {name} not available: {e}")


class NPUKernelsConverter(BaseNPUConverter):
    """All NPU Optimization"""
    kernel_type = None
    kernel_name = "All NPU Kernels"


class NPURMSNormConverter(BaseNPUConverter):
    """RMSNorm NPU Converter"""
    kernel_type = KernelType.RMS_NORM
    kernel_name = "RMSNorm"


class NPURoPEConverter(BaseNPUConverter):
    """RoPE NPU Converter"""
    kernel_type = KernelType.ROPE
    kernel_name = "RoPE"


class NPUPermuteConverter(BaseNPUConverter):
    """PERMUTE NPU Converter"""
    kernel_type = KernelType.PERMUTE
    kernel_name = "Permute"


class NPUDSAConverter(BaseNPUConverter):
    """DSA NPU Converter"""
    kernel_type = KernelType.DSA
    kernel_name = "DSA"


class NPUGMMConverter(BaseNPUConverter):
    """PERMUTE NPU Converter"""
    kernel_type = KernelType.GMM
    kernel_name = "Gmm"
    

class NPUBypassTritionCodegenConverter(BaseNPUConverter):
    """
    apply compile NPU converter
    disable triton fusion
    """
    kernel_type = KernelType.BypassTritionCodegen
    kernel_name = "BypassTritionCodegen"


class NPUFusionAttentionConverter(BaseNPUConverter):
    """FusionAttention NPU Converter"""
    kernel_type = KernelType.FUSIONATTEN
    kernel_name = "FusionAtten"


_registered = False


def register_npu_converters() -> None:
    """
    Register NPU converter to torchtitan
    Enable config in toml file:
        model:
          converters: ["npu"]  # All optimizations
        model:
          converters: ["npu_rms_norm", "npu_rope"]  # Named optimizations
    """
    global _registered

    if _registered:
        return

    register_model_converter(NPUKernelsConverter, "npu")

    _converters = [
        (NPURMSNormConverter, "npu_rms_norm"),
        (NPURoPEConverter, "npu_rope"),
        (NPUPermuteConverter, "npu_permute"),
        (NPUGMMConverter, "npu_gmm"),
        (NPUDSAConverter, "npu_dsa"),
        (NPUFusionAttentionConverter, "npu_fusion_attention"),
        (NPUBypassTritionCodegenConverter, "npu_bypass_triton_codegen")
    ]

    for converter_cls, name in _converters:
        try:
            register_model_converter(converter_cls, name)
        except AssertionError:
            pass

    _registered = True
    logger.info("[NPU] Registered NPU model converters to torchtitan")


register_npu_converters()