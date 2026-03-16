# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

__all__ = [
    "registry",
    "register_npu_converter",
    "ConverterRegistry",
    "BaseConverter",
    "NPUConverter",
]

import importlib
import pkgutil
from pathlib import Path

from .base_converter import BaseConverter
from .npu_converter import NPUConverter

from .registry import ConverterRegistry, register_npu_converter, registry


def _auto_search_conveter():
    package_dir = Path(__file__).parent

    for subdir in ["kernels", "features"]:
        subdir_path = package_dir / subdir
        if subdir_path.exists():
            for _, name, _ in pkgutil.iter_modules([str(subdir_path)]):
                importlib.import_module(f".{subdir}.{name}", package=__package__)


_auto_search_conveter()
