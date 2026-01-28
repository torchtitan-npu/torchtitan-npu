# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

setup(
    name="torchtitan-npu",
    version="0.1.0",
    packages=find_packages(include=['torchtitan_npu']),
    include_package_data=True,
    description="Torchtitan-NPU"
)