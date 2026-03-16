# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

from abc import ABC, abstractmethod
from typing import Dict, Set, Type

import torch.nn as nn
from torchtitan.config.job_config import JobConfig


class BaseConverter(ABC):

    MODEL_IMPL: Dict[str, Type[nn.Module]] = {}
    SUPPORTED_MODELS: Set[str] = {"*"}

    @classmethod
    def get_impl_cls(cls, model_name: str) -> Type[nn.Module]:
        for key, impl_cls in cls.MODEL_IMPL.items():
            if key != "_default" and key in model_name:
                return impl_cls
        return cls.MODEL_IMPL.get("_default")

    @classmethod
    def is_compatible(cls, job_config: JobConfig, model_name: str) -> Type[nn.Module]:
        if "*" in cls.SUPPORTED_MODELS or model_name in cls.SUPPORTED_MODELS:
            return True
        return False

    @classmethod
    @abstractmethod
    def apply(cls, model: nn.Module, model_name: str, **kwargs) -> int:
        pass
