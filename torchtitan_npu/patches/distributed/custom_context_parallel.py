# Adapted from
# https://github.com/pytorch/pytorch/blob/v2.9.0/torch/distributed/tensor/experimental/_attention.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates
# Developed by Huawei Technologies Co., Ltd. based on Meta Platforms, Inc. and affiliates TorchTitan

from typing import Generator, List, Optional, Set

import torch
import torch.nn as nn
import torchtitan
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.experimental._context_parallel._attention import (
    _context_parallel_buffers,
)


_original_create_cp_ctx = torchtitan.distributed.utils.create_context_parallel_ctx


class CustomContextParallelContext:
    """
    A base class providing basic Context Parallelism (CP) capabilities.

    This class is designed to manage tensor operations within a distributed environment
    defined by a DeviceMesh. Upon entering the context, it shards specified tensors
    along a given sequence dimension. Upon exiting, it restores the specified tensors
    to their original state. This class does not perform any additional operations.
    """

    def __init__(
        self,
        mesh: DeviceMesh,
        *,
        buffers: Optional[List[torch.Tensor]] = None,
        buffer_seq_dims: Optional[List[int]] = None,
        no_restore_buffers: Optional[Set[torch.Tensor]] = None,
        load_balance: bool = False
    ):
        self.mesh = mesh
        self.buffers = [] if buffers is None else buffers
        self.buffer_seq_dims = [] if buffer_seq_dims is None else buffer_seq_dims
        self.no_restore_buffers = (
            set() if no_restore_buffers is None else no_restore_buffers
        )
        self.load_balance = load_balance

        if len(self.buffers) != len(self.buffer_seq_dims):
            raise ValueError(
                "`seq_dims` must have the same number of elements as `buffers`."
            )

        for buffer in self.no_restore_buffers:
            if not any(b is buffer for b in self.buffers):
                raise ValueError("`no_restore_buffers` must be a subset of `buffers`.")

        self.original_buffers = []

    @torch.no_grad()
    def __enter__(self):
        # slice input tensors on sequence dim
        self.original_buffers = [
            None if b in self.no_restore_buffers else b.clone() for b in self.buffers
        ]

        if len(self.buffers) > 0:
            device = self.buffers[0].device
            seq_length = self.buffers[0].shape[self.buffer_seq_dims[0]]
            cp_world_size = self.mesh.size()

            if self.load_balance:
                raise NotImplementedError(
                    "Load balance for context parallel is not supported now."
                )
            else:
                load_balance_indices = None

            shards = _context_parallel_buffers(
                self.mesh, self.buffers, self.buffer_seq_dims, load_balance_indices
            )

            for buffer, shard in zip(self.buffers, shards):
                shard_clone = shard.clone()
                buffer.resize_(shard_clone.shape)
                buffer.copy_(shard_clone)

    @torch.no_grad()
    def __exit__(self, exc_type, exc_val, exc_tb):
        # restore specified tensors
        for buffer, original_buffer in zip(self.buffers, self.original_buffers):
            if original_buffer is not None:
                buffer.resize_(original_buffer.shape)
                buffer.copy_(original_buffer)
