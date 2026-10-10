# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/cc286a63599e42480a07928cc362e514ae448a85/tests/integration_tests/__init__.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "OverrideDefinitions",
]


@dataclass
class OverrideDefinitions:
    """
    This class is used to define the override definitions for the integration tests.
    """

    override_args: Sequence[Sequence[str]] = ((" ",),)
    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False
    skip_rocm_test: bool = False
    timeout: int | None = None
    nnodes: int = 1
    env_vars: Mapping[str, str] | None = None
    use_golden: bool = True
    check_loss: bool = True
    # Model-specific launchers may provide their own dataset/assets defaults.
    train_script: str = "scripts/run_train.sh"
    train_args: Sequence[str] | None = None
    expected_steps: Sequence[Sequence[int]] | None = None
    check_resume: bool = False
    verify_ema_checkpoint: bool = False

    def __repr__(self):
        return self.test_descr
