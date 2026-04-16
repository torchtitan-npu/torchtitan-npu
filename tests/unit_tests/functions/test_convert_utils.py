# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for convert_utils module.

These tests verify the utility functions for module/function/method replacement.
"""

import torch.nn as nn

from torchtitan_npu.converters.convert_utils import (
    find_functions,
    find_methods,
    find_modules,
    replace_functions,
    replace_methods,
    replace_modules,
)


class SimpleModule(nn.Module):
    """Simple test module."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.conv = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        return self.linear(x)


def test_find_linear_modules():
    model = SimpleModule()
    matches = find_modules(model, r"Linear")

    assert len(matches) == 1
    assert isinstance(matches[0].module, nn.Linear)
    assert matches[0].attr_name == "linear"


def test_find_conv_modules():
    model = SimpleModule()
    matches = find_modules(model, r"Conv2d")

    assert len(matches) == 1
    assert isinstance(matches[0].module, nn.Conv2d)


def test_find_no_matches():
    model = SimpleModule()
    matches = find_modules(model, r"NonExistent")

    assert len(matches) == 0


def test_module_match_is_meta():
    model = SimpleModule()
    matches = find_modules(model, r"Linear")

    assert matches[0].is_meta is False


def test_replace_module():
    model = SimpleModule()
    matches = find_modules(model, r"Linear")

    new_linear = nn.Linear(10, 20)
    matches[0].replace(new_linear)

    assert model.linear is new_linear


def test_find_functions_in_package():
    import torchtitan_npu.converters.convert_utils as utils_module  # noqa: F401

    matches = find_functions("find_modules", package="torchtitan_npu.converters")

    assert len(matches) >= 1
    assert matches[0].func_name == "find_modules"


def test_find_methods_in_class():
    import tests.unit_tests.functions.test_convert_utils  # noqa: F401

    matches = find_methods(
        "SimpleModule",
        "forward",
        package="tests.unit_tests.functions.test_convert_utils",
    )

    assert len(matches) == 1
    assert matches[0].class_name == "SimpleModule"
    assert matches[0].method_name == "forward"


def test_replace_linear_modules():
    model = SimpleModule()

    def factory(old_module):
        return nn.Linear(10, 10)

    count = replace_modules(model, r"Linear", factory)

    assert count == 1


def test_replace_returns_count():
    model = SimpleModule()

    count = replace_modules(model, r"Linear", lambda m: nn.Linear(10, 10))

    assert count == 1


def test_replace_functions_replaces_loaded_function():
    import torchtitan_npu.converters.convert_utils as utils_module

    original = utils_module.find_modules

    def replacement(*args, **kwargs):
        return []

    count = replace_functions(
        "find_modules", replacement, package="torchtitan_npu.converters"
    )

    assert count >= 1
    assert utils_module.find_modules is replacement

    utils_module.find_modules = original


def test_replace_methods_replaces_loaded_method():
    import tests.unit_tests.functions.test_convert_utils as test_module

    original = test_module.SimpleModule.forward

    def replacement(self, x):
        return x

    count = replace_methods(
        "SimpleModule",
        "forward",
        replacement,
        package="tests.unit_tests.functions.test_convert_utils",
    )

    assert count == 1
    assert test_module.SimpleModule.forward is replacement

    test_module.SimpleModule.forward = original
