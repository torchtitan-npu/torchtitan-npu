# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torchtitan_npu.converters.base_converter import BaseConverter


class MockImplDefault(nn.Module):
    pass


class MockImplDeepseek(nn.Module):
    pass


class MockImplLlama(nn.Module):
    pass


class TestConverter(BaseConverter):
    MODEL_IMPL = {
        "deepseek": MockImplDeepseek,
        "llama": MockImplLlama,
        "_default": MockImplDefault,
    }

    @classmethod
    def apply(cls, model, model_name, **kargs):
        return 0


def test_get_impl_cls_match():
    assert TestConverter.get_impl_cls("deepseek") == MockImplDeepseek
    assert TestConverter.get_impl_cls("llama") == MockImplLlama


def test_get_impl_cls_default():
    assert TestConverter.get_impl_cls("not_impl") == MockImplDefault