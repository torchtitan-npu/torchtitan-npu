# Adapted from
# https://github.com/pytorch/pytorch/blob/v2.10.0/torch/testing/_internal/distributed/fake_pg.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.

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
        common_opts.group_rank, common_opts.group_size, backend_opts
    )


dist.Backend.register_backend(
    dist.Backend.FAKE,
    _create_fake_pg,
    extended_api=True,
    devices=["cpu", "npu"],
)
