# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import runpy
import torchtitan_npu

runpy.run_module("torchtitan.train", run_name="__main__")