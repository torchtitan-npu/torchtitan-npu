# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from tests.conftest import assert_tensor_finite
from tests.smoke_tests.features._dsa_model_helpers import build_model_backed_dsa_inputs

from torchtitan_npu.converters.kernels.dsa import SparseLightningIndexerKLLoss


pytestmark = pytest.mark.smoke


def test_li_loss_forward(npu_device):
    li_loss_fn = SparseLightningIndexerKLLoss()
    loss = li_loss_fn(
        **build_model_backed_dsa_inputs(
            npu_device, batch_size=1, seq_len=2048, requires_grad=False
        )
    )

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert_tensor_finite(loss)


def test_li_loss_backward(npu_device):
    li_loss_fn = SparseLightningIndexerKLLoss()
    inputs = build_model_backed_dsa_inputs(
        npu_device, batch_size=1, seq_len=2048, requires_grad=True
    )

    loss = li_loss_fn(**inputs)
    loss.backward()

    assert inputs["query_indexer"].grad is not None
    assert inputs["key_indexer"].grad is not None
    assert inputs["weights"].grad is not None


def test_sparse_indexer_grad_kl_loss(npu_device):
    import torch_npu

    inputs = build_model_backed_dsa_inputs(
        npu_device, batch_size=1, seq_len=2048, requires_grad=False
    )
    (
        d_query_index,
        d_key_index,
        d_weights,
        loss,
    ) = torch_npu.npu_sparse_lightning_indexer_grad_kl_loss(
        inputs["query"],
        inputs["key"],
        inputs["query_indexer"],
        inputs["key_indexer"],
        inputs["weights"],
        inputs["topk_indices"],
        inputs["softmax_max"],
        inputs["softmax_sum"],
        1.0,
        query_rope=inputs["query_rope"],
        key_rope=inputs["key_rope"],
    )

    assert d_query_index.shape == inputs["query_indexer"].shape
    assert d_key_index.shape == inputs["key_indexer"].shape
    assert d_weights.shape == inputs["weights"].shape
    assert loss.numel() == 1
