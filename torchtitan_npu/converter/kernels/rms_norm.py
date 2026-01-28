# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.nn as nn
import torch_npu

from ..registry import BaseKernel, KernelType, replace_modules

logger = logging.getLogger(__name__)


class NPURMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.dim = dim
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_rms_norm(x, self.weight, self.eps)[0]

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)


def _get_eps(module: nn.Module, default: float = 1e-6) -> float:
    """Securely obtain EPS value"""
    for attr_name in ["eps", "variance_epsilon", "epsilon"]:
        eps = getattr(module, attr_name, None)
        if eps is not None:
            return float(eps)
    return default


def _create_npu_rms_norm(old: nn.Module) -> nn.Module:
    """create NPURMSNorm"""
    dim = old.weight.shape[0]
    eps = _get_eps(old)
    new = NPURMSNorm(dim, eps=eps)
    return new


class RMSNormKernel(BaseKernel):
    kernel_type = KernelType.RMS_NORM

    @classmethod
    def apply(cls, model: nn.Module, **kwargs) -> nn.Module:
        count = replace_modules(model, r"RMSNorm", _create_npu_rms_norm)
        logger.info(f"  Replaced {count} RMSNorm module(s)")
        return model