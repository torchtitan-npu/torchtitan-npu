# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

pytestmark = pytest.mark.smoke


def test_moe_token_permute_forward(npu_device):
    import torch_npu

    x = torch.randn(8, 16, dtype=torch.float32, device=npu_device)
    indices = torch.tensor([[0, 1]] * 8, dtype=torch.int32, device=npu_device)

    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x, indices)

    assert routed_input.shape[1] == x.shape[1]
    assert sorted_indices.numel() >= x.shape[0]


def test_moe_token_unpermute_restores_shape(npu_device):
    import torch_npu

    x = torch.randn(8, 16, dtype=torch.float32, device=npu_device)
    indices = torch.tensor([[0, 1]] * 8, dtype=torch.int32, device=npu_device)
    routed_input, sorted_indices = torch_npu.npu_moe_token_permute(x, indices)
    top_scores = torch.ones(8, 2, dtype=torch.float32, device=npu_device)

    restored = torch_npu.npu_moe_token_unpermute(
        routed_input, sorted_indices, top_scores
    )

    assert restored.shape == x.shape
