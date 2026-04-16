# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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


def test_is_compatible_supports_wildcard_models():
    assert TestConverter.is_compatible(job_config=None, model_name="anything")


class RestrictedConverter(BaseConverter):
    MODEL_IMPL = {"_default": MockImplDefault}
    SUPPORTED_MODELS = {"llama3"}

    @classmethod
    def apply(cls, model, model_name, **kwargs):
        return 0


def test_is_compatible_rejects_unsupported_model():
    assert not RestrictedConverter.is_compatible(
        job_config=None, model_name="deepseek_v3"
    )
