# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def test_quant_converter_replace():
    from torchtitan.components.quantization.mx import (
        MXGroupedMMConverter,
        MXLinearConverter,
    )

    from torchtitan_npu.converters import quant_converter

    assert MXLinearConverter.__init__ == quant_converter.npu_quant_linear_converter_init
    assert MXLinearConverter.convert == quant_converter.npu_quant_linear_converter
    assert (
        MXGroupedMMConverter.__init__
        == quant_converter.npu_quant_grouped_mm_converter_init
    )
    assert (
        MXGroupedMMConverter.convert == quant_converter.npu_quant_grouped_mm_converter
    )
