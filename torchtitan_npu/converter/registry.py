# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import inspect
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Callable, Type, List

import torch.nn as nn

logger = logging.getLogger(__name__)


class KernelType(Enum):
    RMS_NORM = auto()
    ROPE = auto()
    PERMUTE = auto()
    GMM = auto()
    DSA = auto()
    FUSIONATTEN = auto()
    BypassTritionCodegen = auto()


class KernelRegistry:
    """Kernel registration"""
    _instance: Optional["KernelRegistry"] = None
    _initialized: bool = False

    def __new__(cls) -> "KernelRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._registry: dict[KernelType, Type["BaseKernel"]] = {}
        self._initialized = True

    def register(self, kernel_type: KernelType, kernel_cls: Type["BaseKernel"]) -> None:
        if kernel_type in self._registry:
            logger.warning(f"Overwriting kernel: {kernel_type.name}")
        self._registry[kernel_type] = kernel_cls
        logger.info(f"Registered: {kernel_type.name} -> {kernel_cls.__name__}")

    def get(self, kernel_type: KernelType) -> Optional[Type["BaseKernel"]]:
        return self._registry.get(kernel_type)

    def get_all(self) -> list[Type["BaseKernel"]]:
        return list(self._registry.values())

    def clear(self) -> None:
        self._registry.clear()


KERNEL_REGISTRY = KernelRegistry()


class BaseKernel(ABC):
    """
    Kernel Base Class

    When a subclass defines kernel_type, it is automatically registered to KERNEL_REGISTRY.
    """
    kernel_type: Optional[KernelType] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.kernel_type is not None:
            KERNEL_REGISTRY.register(cls.kernel_type, cls)

    @classmethod
    @abstractmethod
    def apply(cls, model: nn.Module, **kwargs) -> nn.Module:
        raise NotImplementedError


@dataclass
class ModuleMatch:
    parent: nn.Module
    attr_name: str
    module: nn.Module
    full_name: str

    @property
    def is_meta(self) -> bool:
        p = next(self.module.parameters(), None)
        return p is not None and p.device.type == "meta"

    def replace(self, new_module: nn.Module, log: bool = False) -> None:
        setattr(self.parent, self.attr_name, new_module)
        if log:
            logger.info(f"   {self.full_name}")


@dataclass
class FunctionMatch:
    module_path: str
    func_name: str
    func: Callable

    @property
    def full_path(self) -> str:
        return f"{self.module_path}.{self.func_name}"

    def replace(self, new_func: Callable, log: bool = False) -> None:
        if mod := sys.modules.get(self.module_path):
            setattr(mod, self.func_name, new_func)
            if log:
                logger.info(f"   {self.full_path}")


@dataclass
class MethodMatch:
    module_path: str
    class_name: str
    method_name: str
    cls: type
    method: Callable

    @property
    def full_path(self) -> str:
        return f"{self.module_path}.{self.class_name}.{self.method_name}"

    def replace(self, new_method: Callable, log: bool = False) -> None:
        setattr(self.cls, self.method_name, new_method)
        if log:
            logger.info(f"   {self.full_path}")


def _get_package(model: Optional[nn.Module] = None, package: Optional[str] = None) -> str:
    if package:
        return package
    if model:
        return model.__class__.__module__.rsplit(".", 1)[0]
    raise ValueError("Must provide either model or package")


def find_modules(model: nn.Module, pattern: str) -> List[ModuleMatch]:
    regex = re.compile(pattern)
    return [
        ModuleMatch(
            model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model,
            name.rsplit(".", 1)[-1] if "." in name else name,
            module,
            name
        )
        for name, module in model.named_modules()
        if regex.search(module.__class__.__name__) and name
    ]


def find_functions(
    func_name: str,
    model: Optional[nn.Module] = None,
    package: Optional[str] = None,
) -> List[FunctionMatch]:
    pkg = _get_package(model, package)
    result = []
    for path, mod in sys.modules.items():
        if not mod or not path.startswith(pkg):
            continue
        func = getattr(mod, func_name, None)
        if callable(func) and not isinstance(func, type):
            result.append(FunctionMatch(path, func_name, func))
    return result


def find_methods(
    class_name: str,
    method_name: str,
    model: Optional[nn.Module] = None,
    package: Optional[str] = None,
) -> List[MethodMatch]:
    pkg = _get_package(model, package)
    matches = []

    for mod_path, mod in sys.modules.items():
        if mod is None or not mod_path.startswith(pkg):
            continue

        cls = getattr(mod, class_name, None)
        if (inspect.isclass(cls)
                and cls.__module__ == mod_path
                and hasattr(cls, method_name)):
            matches.append(MethodMatch(
                module_path=mod_path,
                class_name=class_name,
                method_name=method_name,
                cls=cls,
                method=getattr(cls, method_name)
            ))

    return matches


def replace_modules(model: nn.Module, pattern: str, factory: Callable) -> int:
    matches = find_modules(model, pattern)
    for m in matches:
        try:
            m.replace(factory(m.module))
        except Exception as e:
            logger.error(f"  ✗ {m.full_name}: {e}")
    return len(matches)


def replace_functions(
    func_name: str,
    new_func: Callable,
    model: Optional[nn.Module] = None,
    package: Optional[str] = None,
) -> int:
    matches = find_functions(func_name, model=model, package=package)
    for m in matches:
        m.replace(new_func)
    return len(matches)


def replace_methods(
    class_name: str,
    method_name: str,
    new_method: Callable,
    model: Optional[nn.Module] = None,
    package: Optional[str] = None,
) -> int:
    matches = find_methods(class_name, method_name, model=model, package=package)
    for m in matches:
        m.replace(new_method)
    return len(matches)
