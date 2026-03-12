# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch
from torchtitan_npu.converters.npu_converter import NPUConverter
from torchtitan_npu.converters.base_converter import BaseConverter


def test_converter_init(mock_job_config, mock_parallel_dims):
    config = mock_job_config(model_name="llama3")
    converter = NPUConverter(config, mock_parallel_dims)
    assert converter.model_name == "llama3"
    assert converter.job_config == config


def test_convert_calls_patch_apply(mock_job_config, mock_parallel_dims, simple_model):
    config = mock_job_config(model_name="llama3")

    mock_patch_cls = MagicMock(spec=BaseConverter)
    mock_patch_cls.apply.return_value = 5
    mock_patch_cls.is_compatible.return_value = True

    # Use patch.object to mock class attributes without accessing private attributes directly
    with patch.object(NPUConverter, '_patch_cls', mock_patch_cls), \
            patch.object(NPUConverter, '_patch_name', 'test_patch'), \
            patch.object(NPUConverter, '_supported_models', {'*'}):
        converter = NPUConverter(config, mock_parallel_dims)
        converter.convert(simple_model)

    mock_patch_cls.apply.assert_called_once_with(simple_model, "llama3")


def test_convert_returns_model(mock_job_config, mock_parallel_dims, simple_model):
    config = mock_job_config(model_name="llama3")

    mock_patch_cls = MagicMock(spec=BaseConverter)
    mock_patch_cls.apply.return_value = 0
    mock_patch_cls.is_compatible.return_value = True

    with patch.object(NPUConverter, '_patch_cls', mock_patch_cls), \
            patch.object(NPUConverter, '_patch_name', 'test_patch'), \
            patch.object(NPUConverter, '_supported_models', {'*'}):
        converter = NPUConverter(config, mock_parallel_dims)
        result = converter.convert(simple_model)

    assert result is simple_model