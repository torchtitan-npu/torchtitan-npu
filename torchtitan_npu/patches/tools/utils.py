# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib


def load_class_from_string(class_path: str):
    """Dynamically load class according to a string."""
    try:
        module_path, class_name = class_path.rsplit('.', 1)
    except ValueError as e:
        raise ValueError(f"Class string path error: '{class_path}', need to be 'module.path.ClassName'") from e

    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}") from e

    try:
        cls = getattr(module, class_name)
    except AttributeError as e:
        raise AttributeError(f"Module '{module_path}' does not have class '{class_name}'") from e

    return cls