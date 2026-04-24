# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

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

from torchtitan_npu.converters.kernels.deepseek_v4_sfa import li_loss_adapter


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

    def test_li_loss_adapter_data_prep(self):
        mock_fused_op = _mock_fused_fn
        mock_fused_op.reset_mock()
        mock_fused_op.return_value = torch.tensor(0.5)

        result = li_loss_adapter(
            self.mock_self,
            self.q,
            self.kv_compress,
            self.q_indexer,
            self.k_indexer,
            self.weights,
            self.compress_topk_idxs,
            self.index_score,
            None,
            self.offset,
        )

        args, kwargs = mock_fused_op.call_args
        # q: [B,S,N,D] -> transpose(0,1) -> [S,B,N,D], bf16
        self.assertEqual(
            args[0].shape, (self.seq_len, self.batch_size, self.n_heads, self.head_dim)
        )
        self.assertEqual(args[0].dtype, torch.bfloat16)
        # kv_compress: [B,S_c,D] -> transpose(0,1) -> [S_c,B,D], bf16
        self.assertEqual(
            args[1].shape, (self.seq_len_compress, self.batch_size, self.head_dim)
        )
        self.assertEqual(args[1].dtype, torch.bfloat16)
        # q_indexer: [B,S,N_idx,D] -> transpose(0,1) -> [S,B,N_idx,D], bf16
        self.assertEqual(
            args[2].shape,
            (self.seq_len, self.batch_size, self.n_idx_heads, self.idx_dim),
        )
        self.assertEqual(args[2].dtype, torch.bfloat16)
        # k_indexer: [B,S_c,D] -> unsqueeze(2) -> [B,S_c,1,D] -> transpose(0,1) -> [S_c,B,1,D], bf16
        self.assertEqual(
            args[3].shape, (self.seq_len_compress, self.batch_size, 1, self.idx_dim)
        )
        self.assertEqual(args[3].dtype, torch.bfloat16)
        # weights: [B,S,N_idx] -> transpose(0,1) -> [S,B,N_idx], bf16
        self.assertEqual(
            args[4].shape, (self.seq_len, self.batch_size, self.n_idx_heads)
        )
        self.assertEqual(args[4].dtype, torch.bfloat16)
        # compress_topk_idxs: int32, -1 should stay -1
        self.assertEqual(args[5].dtype, torch.int32)
        self.assertEqual(args[5][0, 0, 0].item(), -1)
        # kwargs passed to fused op
        self.assertEqual(kwargs["scale_value"], self.softmax_scale)
        self.assertEqual(kwargs["cmp_ratio"], self.compress_ratio)

        self.assertEqual(result.item(), 0.5)


if __name__ == "__main__":
    unittest.main()
