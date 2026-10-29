# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the BSD-style license found in the LICENSE file.

"""Tiny NPU collective used to verify Lite Actions topology without model loading."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401 - registers the HCCL backend


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    expected_world = int(os.environ["LITE_SMOKE_EXPECTED_WORLD"])
    if world_size != expected_world:
        raise RuntimeError(f"expected world_size={expected_world}, got {world_size}")

    torch.npu.set_device(local_rank)
    dist.init_process_group(backend="hccl", init_method="env://", timeout=__import__(
        "datetime").timedelta(seconds=45))
    try:
        # Explicitly construct the DP domain over all launched ranks.
        dp_ranks = list(range(world_size))
        dp_group = dist.new_group(ranks=dp_ranks, backend="hccl")
        value = torch.tensor([rank], dtype=torch.int32, device=f"npu:{local_rank}")
        dist.all_reduce(value, op=dist.ReduceOp.SUM, group=dp_group)
        expected_sum = world_size * (world_size - 1) // 2
        if int(value.cpu().item()) != expected_sum:
            raise RuntimeError(
                f"DP all_reduce mismatch: expected {expected_sum}, got {int(value.cpu().item())}")
        dist.barrier(group=dp_group)
        if rank == 0:
            print("DISPATCH_SMOKE_PASS " + json.dumps({
                "world_size": world_size,
                "dp_world_size": world_size,
                "all_reduce_sum": expected_sum,
            }, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
