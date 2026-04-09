# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base_converter import BaseConverter


@dataclass
class PatchInfo:
    name: str
    patch_cls: type["BaseConverter"]
    supported_models: set[str] = field(default_factory=lambda: {"*"})


class ConverterRegistry:
    _instance = None
    _patches: dict[str, PatchInfo]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._patches = {}
        return cls._instance

    @staticmethod
    def _register_as_model_converter(
        name: str,
        patch_cls: type["BaseConverter"],
        supported_models: set[str],
    ):
        from torchtitan.protocols.model_converter import register_model_converter

        from .npu_converter import NPUConverter

        converter_cls = type(
            f"{patch_cls.__name__}Converter",
            (NPUConverter,),
            {
                "_patch_cls": patch_cls,
                "_patch_name": name,
                "_supported_models": supported_models,
            },
        )

        register_model_converter(converter_cls, name)

    def register(self, name: str, supported_models: set[str] | None = None):
        def decorator(patch_cls: type["BaseConverter"]):
            models = supported_models
            if models is None:
                models = getattr(patch_cls, "SUPPORTED_MODELS", {"*"})

            self._patches[name] = PatchInfo(
                name=name, patch_cls=patch_cls, supported_models=models
            )

            self._register_as_model_converter(name, patch_cls, models)

            return patch_cls

        return decorator

    def get(self, name: str) -> PatchInfo | None:
        return self._patches.get(name)


registry = ConverterRegistry()


def register_npu_converter(name: str, supported_models: set[str] | None = None):
    return registry.register(name, supported_models)
