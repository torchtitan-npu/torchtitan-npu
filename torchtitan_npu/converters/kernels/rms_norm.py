# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch_npu

from ..base_converter import BaseConverter
from ..convert_utils import replace_modules
from ..registry import register_npu_converter

logger = logging.getLogger(__name__)


class NPURMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = None):
        super().__init__()
        self.dim = dim
        self.eps = float(eps) if eps is not None else None
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Matches the default implementation of nn.RMSNorm:
        # - Use user-provided eps if it exists.
        # - Otherwise, use the machine epsilon of the current input `x`.
        resolved_eps = self.eps if self.eps is not None else torch.finfo(x.dtype).eps
        return torch_npu.npu_rms_norm(x, self.weight, resolved_eps)[0]

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'


def _get_eps(module: nn.Module) -> Optional[float]:
    for attr_name in ["eps", "variance_epsilon", "epsilon"]:
        eps = getattr(module, attr_name, None)
        if eps is not None:
            return float(eps)
    return None


def _create_npu_rms_norm(old: nn.Module) -> nn.Module:
    dim = old.weight.shape[-1]
    eps = _get_eps(old)
    new = NPURMSNorm(dim, eps=eps)
    return new


@register_npu_converter("npu_rms_norm")
class RMSNormKernel(BaseConverter):

    @classmethod
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> nn.Module:
        count = replace_modules(model, r"RMSNorm", _create_npu_rms_norm)
        return count