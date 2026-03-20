# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from types import SimpleNamespace

import torch.nn as nn
import pytest

from torchtitan_npu.converters.base_converter import BaseConverter
from torchtitan_npu.converters.npu_converter import NPUConverter
from torchtitan_npu.converters.quant_converter import module_filter_fn


class _RejectingPatch(BaseConverter):
    SUPPORTED_MODELS = {"deepseek_v3"}

    @classmethod
    def apply(cls, model, model_name, **kwargs):
        return 0


class RejectingConverter(NPUConverter):
    _patch_cls = _RejectingPatch
    _patch_name = "rejecting_patch"
    _supported_models = _RejectingPatch.SUPPORTED_MODELS


def test_module_filter_fn_accepts_unfiltered_linear():
    mod = nn.Linear(8, 16)

    assert module_filter_fn(mod, "model.layers.0.mlp.w1", ["attention"])


def test_module_filter_fn_rejects_non_linear_module():
    mod = nn.ReLU()

    assert not module_filter_fn(mod, "model.layers.0.relu", [])


def test_module_filter_fn_rejects_filtered_fqn():
    mod = nn.Linear(8, 16)

    assert not module_filter_fn(mod, "model.layers.0.attention.proj", ["attention"])


def test_npu_converter_convert_raises_for_unsupported_model():
    job_config = SimpleNamespace(model=SimpleNamespace(name="llama3"))
    converter = RejectingConverter(job_config, parallel_dims=object())

    with pytest.raises(ValueError, match="NOT compatible"):
        converter.convert(nn.Linear(8, 16))

