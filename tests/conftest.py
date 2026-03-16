# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Pytest configuration for torchtitan-npu tests.

This conftest ensures that torchtitan_npu patches are applied before
running any tests, including torchtitan upstream tests.
"""

import pytest


def pytest_configure(config):
    """
    Called before test collection and execution.
    Import torchtitan_npu to apply all NPU patches.
    """
    import torchtitan_npu  # noqa: F401
