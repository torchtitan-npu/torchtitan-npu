# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from tests.conftest import assert_tensor_finite, stable_randn
from tests.smoke_tests.features._dsa_model_helpers import (
    build_model_backed_dsa_inputs,
    run_lightning_indexer_smoke,
)


pytestmark = pytest.mark.smoke


def test_dsa_enabled(npu_device):
    topk_indices, scores = run_lightning_indexer_smoke(npu_device)

    assert topk_indices.shape[:2] == (1, 128)
    assert topk_indices.dtype == torch.int32
    assert scores is not None


def test_li_loss_enabled(npu_device):
    from torchtitan_npu.converters.kernels.dsa import SparseLightningIndexerKLLoss

    li_loss_fn = SparseLightningIndexerKLLoss()
    loss = li_loss_fn(
        **build_model_backed_dsa_inputs(
            npu_device, batch_size=1, seq_len=2048, requires_grad=False
        )
    )

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert_tensor_finite(loss)


def test_gmm_enabled(npu_device):
    import torch_npu

    total_tokens = 64
    hidden_dim = 256
    intermediate_dim = 512
    num_experts = 8

    output = torch_npu.npu_grouped_matmul(
        [
            stable_randn(
                total_tokens, hidden_dim, dtype=torch.bfloat16, device=npu_device
            )
        ],
        [
            stable_randn(
                num_experts,
                hidden_dim,
                intermediate_dim,
                dtype=torch.bfloat16,
                device=npu_device,
            )
        ],
        bias=None,
        group_list=torch.tensor(
            [8] * num_experts, dtype=torch.int64, device=npu_device
        ),
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]

    assert output.shape == (total_tokens, intermediate_dim)
    assert_tensor_finite(output)
