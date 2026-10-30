# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Named OS probes; commands are implemented by the trusted dispatcher."""
from tests.integration_tests import OverrideDefinitions

def build_test_list():
    return [OverrideDefinitions(test_name=f"worker_os_{worker}", test_descr=f"OS probe on {worker}",
        ci_kind="worker_probe", ci_overrides={"worker":worker}, nnodes=0, ngpu=0, timeout=90)
        for worker in ("blue", "yellow")]
