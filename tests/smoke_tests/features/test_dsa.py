# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from tests.conftest import assert_tensor_finite
from tests.smoke_tests.features._dsa_model_helpers import (
    build_model_backed_dsa_inputs,
    run_lightning_indexer_smoke,
)


pytestmark = pytest.mark.smoke


def test_dsa_forward_returns_scalar_finite_loss(npu_device):
    from torchtitan_npu.converters.kernels.dsa import SparseLightningIndexerKLLoss

    li_loss_fn = SparseLightningIndexerKLLoss()
    inputs = build_model_backed_dsa_inputs(npu_device, batch_size=1, seq_len=2048, requires_grad=False)
    loss = li_loss_fn(**inputs)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    assert_tensor_finite(loss)


def test_dsa_backward_produces_finite_gradients(npu_device):
    from torchtitan_npu.converters.kernels.dsa import SparseLightningIndexerKLLoss

    li_loss_fn = SparseLightningIndexerKLLoss()
    inputs = build_model_backed_dsa_inputs(npu_device, batch_size=1, seq_len=2048, requires_grad=True)
    loss = li_loss_fn(**inputs)
    loss.backward()

    assert inputs["query_indexer"].grad is not None
    assert inputs["key_indexer"].grad is not None
    assert inputs["weights"].grad is not None
    assert_tensor_finite(inputs["query_indexer"].grad)
    assert_tensor_finite(inputs["key_indexer"].grad)
    assert_tensor_finite(inputs["weights"].grad)


@pytest.mark.parametrize("batch_size,seq_len", [(1, 2048), (2, 2048)])
def test_dsa_supports_model_backed_shapes(batch_size, seq_len, npu_device):
    from torchtitan_npu.converters.kernels.dsa import SparseLightningIndexerKLLoss

    li_loss_fn = SparseLightningIndexerKLLoss()
    inputs = build_model_backed_dsa_inputs(npu_device, batch_size=batch_size, seq_len=seq_len, requires_grad=False)
    loss = li_loss_fn(**inputs)

    assert_tensor_finite(loss)


def test_lightning_indexer_returns_expected_layout(npu_device):
    topk_indices, scores = run_lightning_indexer_smoke(npu_device)

    assert topk_indices.shape[:2] == (1, 128)
    assert topk_indices.dtype == torch.int32
    assert scores is not None

