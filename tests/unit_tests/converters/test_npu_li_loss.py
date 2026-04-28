# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
import unittest
from unittest.mock import MagicMock

import torch

# Inject fake mindspeed module hierarchy before importing the adapter
_mock_fused_fn = MagicMock()
_mock_ops_mod = types.ModuleType(
    "mindspeed.ops.npu_sparse_lightning_indexer_grad_kl_loss"
)
_mock_ops_mod.npu_sparse_lightning_indexer_grad_kl_loss = _mock_fused_fn
sys.modules.setdefault("mindspeed", types.ModuleType("mindspeed"))
sys.modules.setdefault("mindspeed.ops", types.ModuleType("mindspeed.ops"))
sys.modules.setdefault(
    "mindspeed.ops.npu_sparse_lightning_indexer_grad_kl_loss", _mock_ops_mod
)


class TestLILossKernel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len = 512
        self.seq_len_compress = 128
        self.n_heads = 16
        self.head_dim = 128
        self.n_idx_heads = 4
        self.idx_dim = 64
        self.topk = 64
        self.compress_ratio = 4
        self.softmax_scale = 0.125
        self.offset = 512

        self.q = torch.randn(
            self.batch_size,
            self.seq_len,
            self.n_heads,
            self.head_dim,
            dtype=torch.float32,
        )
        self.kv_compress = torch.randn(
            self.batch_size, self.seq_len_compress, self.head_dim, dtype=torch.float32
        )
        self.q_indexer = torch.randn(
            self.batch_size,
            self.seq_len,
            self.n_idx_heads,
            self.idx_dim,
            dtype=torch.float32,
        )
        self.k_indexer = torch.randn(
            self.batch_size, self.seq_len_compress, self.idx_dim, dtype=torch.float32
        )
        self.weights = torch.randn(
            self.batch_size, self.seq_len, self.n_idx_heads, dtype=torch.float32
        )
        self.compress_topk_idxs = torch.randint(
            0, self.seq_len_compress, (self.batch_size, self.seq_len, self.topk)
        )
        self.compress_topk_idxs[0, 0, 0] = -1
        self.index_score = torch.randn(
            self.batch_size, self.seq_len, self.topk, dtype=torch.float32
        )

        self.mock_self = MagicMock()
        self.mock_self.softmax_scale = self.softmax_scale
        self.mock_self.compress_ratio = self.compress_ratio


if __name__ == "__main__":
    unittest.main()
