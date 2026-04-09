# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file is derived from PyTorch,
# https://github.com/pytorch/pytorch/blob/v2.10.0/torch/testing/_internal/distributed/fake_pg.py
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.distributed as dist
from torch._C._distributed_c10d import FakeProcessGroup


def _create_fake_pg(common_opts, backend_opts):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convenient tool when playing
    with distributed but don't care about the actual data.
    """
    return FakeProcessGroup._create_internal(
        common_opts.group_rank,
        common_opts.group_size,
        # pyrefly: ignore [bad-argument-count]
        backend_opts,
    )


dist.Backend.register_backend(
    # pyrefly: ignore [missing-attribute]
    dist.Backend.FAKE,
    _create_fake_pg,
    extended_api=True,
    devices=["cpu", "npu"],
)
