# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import types
import unittest
from unittest.mock import MagicMock

import torch

from torchtitan_npu.converters.kernels.deepseek_v4_sfa import sdpa_to_sfa_adapter

_mock_fused_fn = MagicMock()


class MockSparseAttn:
    apply = _mock_fused_fn


_mock_ops_mod = types.ModuleType("mindspeed.ops.npu_sparse_attn_shared_kv")

_mock_ops_mod.SparseAttnSharedKV = MockSparseAttn
_mock_ops_mod.npu_sparse_attn_shared_kv = _mock_fused_fn

sys.modules.setdefault("mindspeed", types.ModuleType("mindspeed"))
sys.modules.setdefault("mindspeed.ops", types.ModuleType("mindspeed.ops"))
sys.modules.setdefault("mindspeed.ops.npu_sparse_attn_shared_kv", _mock_ops_mod)


class TestSFANPUKernel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_len_q = 1024
        self.seq_len_kv = 1024
        self.num_heads_q = 16
        self.head_dim = 128
        self.compress_ratio = 4
        self.topk = 256
        self.softmax_scale = 0.125

        self.query = torch.randn(
            self.batch_size, self.seq_len_q, self.num_heads_q, self.head_dim
        )
        self.kv_states = torch.randn(self.seq_len_kv, self.batch_size, self.head_dim)
        self.kv_compress = torch.randn(
            self.seq_len_kv // self.compress_ratio, self.batch_size, self.head_dim
        )
        self.indices = torch.randint(
            0, self.seq_len_kv, (self.batch_size, self.seq_len_q, self.topk)
        ).to(torch.int64)
        self.sinks = torch.randn(1, 128)

    def test_sdpa_to_sfa_adapter_flow(self):
        mock_fused_op = _mock_fused_fn
        mock_fused_op.reset_mock()

        mock_fused_op.return_value = torch.randn(
            self.batch_size, self.seq_len_q, self.num_heads_q, self.head_dim
        )

        mock_self = MagicMock()
        mock_self.softmax_scale = self.softmax_scale
        mock_self.compress_ratio = self.compress_ratio

        output = sdpa_to_sfa_adapter(
            mock_self,
            self.query,
            self.kv_states,
            self.sinks,
            self.kv_compress,
            self.indices,
        )

        args, _ = mock_fused_op.call_args

        cmp_sparse_indices = args[7]
        self.assertEqual(cmp_sparse_indices.dtype, torch.int32)
        ori_kv_processed = args[1]
        self.assertEqual(
            ori_kv_processed.shape, (self.seq_len_kv, self.batch_size, 1, self.head_dim)
        )

        self.assertTrue(output.is_contiguous())


if __name__ == "__main__":
    unittest.main()
