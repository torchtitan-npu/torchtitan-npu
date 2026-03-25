# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

from tests.smoke_tests.conftest import skip_on_runtime_unsupported

from torchtitan_npu.converters.kernels.quant_gmm import (
    GMMFunctionHif8,
    group_size_params,
)

pytestmark = pytest.mark.smoke


def test_quant_linear_mxfp8(npu_device):
    import torch_npu

    x = torch.randn(16, 256, dtype=torch.bfloat16, device=npu_device)
    weight = torch.randn(512, 256, dtype=torch.bfloat16, device=npu_device)

    try:
        x_quant, x_scale = torch_npu.npu_dynamic_mx_quant(
            x, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
        weight_quant, weight_scale = torch_npu.npu_dynamic_mx_quant(
            weight, axis=-1, dst_type=torch.float8_e4m3fn, scale_alg=1
        )
    except RuntimeError as error:
        skip_on_runtime_unsupported(
            error,
            ("does not support opType [DynamicMxQuant]",),
            "DynamicMxQuant is not supported on the current Ascend SOC",
        )

    assert x_quant.shape == x.shape
    assert weight_quant.shape == weight.shape
    assert x_scale is not None
    assert weight_scale is not None


def test_quant_linear_hif8(npu_device):
    import torch_npu

    x = torch.randn(16, 256, dtype=torch.bfloat16, device=npu_device)
    weight = torch.randn(256, 256, dtype=torch.bfloat16, device=npu_device)

    try:
        x_quant, x_scale = torch_npu.npu_dynamic_quant(
            x, dst_type=torch_npu.hifloat8, quant_mode="pertensor"
        )
        weight_quant, weight_scale = torch_npu.npu_dynamic_quant(
            weight, dst_type=torch_npu.hifloat8, quant_mode="pertensor"
        )
    except RuntimeError as error:
        skip_on_runtime_unsupported(
            error,
            ("only support socVersion Ascend950",),
            "HiF8 dynamic quant is only supported on Ascend950-class runtimes",
        )

    assert x_quant.shape == x.shape
    assert weight_quant.shape == weight.shape
    assert x_scale is not None
    assert weight_scale is not None


def test_quant_gmm_hif8_helper_quantizes_tensor_shapes(npu_device):
    import torch_npu

    group_size_params["g_size"] = 4
    x = torch.randn(8, 16, dtype=torch.bfloat16, device=npu_device)
    weight = torch.randn(4, 16, 32, dtype=torch.bfloat16, device=npu_device)

    try:
        x_quant, x_scale, weight_quant, weight_scale = GMMFunctionHif8.quantize(
            x,
            weight,
            torch_npu.hifloat8,
            torch_npu.hifloat8,
        )
    except RuntimeError as error:
        skip_on_runtime_unsupported(
            error,
            ("only support socVersion Ascend950", "should be in dtype support list"),
            "HiF8 grouped quant is not supported on the current runtime or input dtype",
        )

    assert x_quant.shape == x.shape
    assert weight_quant.shape == weight.shape
    assert x_scale is not None
    assert weight_scale is not None
