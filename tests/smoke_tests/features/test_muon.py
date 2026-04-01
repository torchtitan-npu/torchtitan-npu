# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn

from torchtitan_npu.patches.optimizer.muon_optimizer import build_muon_hybrid_optimizers

pytestmark = pytest.mark.smoke


@pytest.mark.smoke
def test_muon_training_loop_updates_params(npu_device, muon_config):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.LayerNorm(64),
    ).to(npu_device)

    optimizer = build_muon_hybrid_optimizers([model], muon_config(), None)

    orig_linear_weight = model[0].weight.data.clone()
    orig_ln_weight = model[1].weight.data.clone()

    x = torch.randn(2, 32, device=npu_device)
    loss = model(x).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()

    assert not torch.equal(model[0].weight.data, orig_linear_weight)
    assert not torch.equal(model[1].weight.data, orig_ln_weight)


@pytest.mark.smoke
def test_muon_zero_grad_works(npu_device, muon_config):
    model = nn.Linear(16, 16).to(npu_device)
    optimizer = build_muon_hybrid_optimizers([model], muon_config(), None)

    x = torch.randn(2, 16, device=npu_device)
    loss = model(x).sum()
    loss.backward()

    assert model.weight.grad is not None

    optimizer.zero_grad()

    assert model.weight.grad is None or torch.all(model.weight.grad == 0)
