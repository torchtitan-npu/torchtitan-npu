# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from types import SimpleNamespace

from torchtitan_npu.converters import quant_converter


@dataclass
class QuantizeCallSpy:
    model: object = None
    config: object = None
    filter_fn: object = None

    def record(self, model, config, filter_fn):
        self.model = model
        self.config = config
        self.filter_fn = filter_fn


def _patch_quant_runtime(monkeypatch, *, linear_recipe=None, grouped_recipe=None):
    monkeypatch.setattr(
        quant_converter,
        "validate_quantization_job_config",
        lambda job_config: None,
    )
    monkeypatch.setattr(quant_converter, "is_a5", lambda: True)
    if linear_recipe is not None:
        monkeypatch.setattr(
            quant_converter.TorchMXLinearConfig,
            "from_recipe_name",
            staticmethod(lambda recipe_name: linear_recipe(recipe_name)),
        )
    if grouped_recipe is not None:
        monkeypatch.setattr(
            quant_converter.TorchMoETrainingConfig,
            "from_recipe_name",
            staticmethod(lambda recipe_name: grouped_recipe(recipe_name)),
        )


def _build_linear_job_config():
    return SimpleNamespace(
        compile=SimpleNamespace(enable=False, components=[]),
        parallelism=SimpleNamespace(tensor_parallel_degree=1),
        quantize=SimpleNamespace(
            linear=SimpleNamespace(
                mx=SimpleNamespace(recipe_name="mxfp8", filter_fqns=["attention"])
            )
        ),
    )


def _build_grouped_mm_job_config():
    return SimpleNamespace(
        quantize=SimpleNamespace(
            grouped_mm=SimpleNamespace(
                mx=SimpleNamespace(recipe_name="mxfp8", fqns=["layers.0.moe"])
            )
        )
    )


def test_quant_converter_replace():
    from torchtitan.components.quantization.mx import MXGroupedMMConverter, MXLinearConverter

    assert MXLinearConverter.__init__ == quant_converter.npu_quant_linear_converter_init
    assert MXLinearConverter.convert == quant_converter.npu_quant_linear_converter
    assert MXGroupedMMConverter.__init__ == quant_converter.npu_quant_grouped_mm_converter_init
    assert MXGroupedMMConverter.convert == quant_converter.npu_quant_grouped_mm_converter


def test_npu_quant_linear_converter_init_sets_runtime_fields(monkeypatch):
    _patch_quant_runtime(
        monkeypatch,
        linear_recipe=lambda recipe_name: {"recipe": recipe_name},
    )

    converter = SimpleNamespace()
    job_config = _build_linear_job_config()

    quant_converter.npu_quant_linear_converter_init(converter, job_config, parallel_dims=object())

    assert converter.enabled is True
    assert converter.filter_fqns == ["attention"]
    assert converter.config == {"recipe": "mxfp8"}


def test_npu_quant_linear_converter_calls_linear_quantize(monkeypatch):
    spy = QuantizeCallSpy()
    monkeypatch.setattr(
        quant_converter,
        "linear_quantize_",
        spy.record,
    )

    converter = SimpleNamespace(enabled=True, config={"recipe": "mxfp8"}, filter_fqns=["attention"])
    model = object()

    quant_converter.npu_quant_linear_converter(converter, model)

    assert spy.model is model
    assert spy.config == {"recipe": "mxfp8"}
    assert spy.filter_fn is not None


def test_npu_quant_grouped_mm_converter_calls_grouped_quantize(monkeypatch):
    _patch_quant_runtime(
        monkeypatch,
        grouped_recipe=lambda recipe_name: {"recipe": recipe_name},
    )

    spy = QuantizeCallSpy()
    monkeypatch.setattr(
        quant_converter,
        "grouped_quantize_",
        spy.record,
    )

    converter = SimpleNamespace()
    job_config = _build_grouped_mm_job_config()

    quant_converter.npu_quant_grouped_mm_converter_init(converter, job_config, parallel_dims=object())

    model = object()
    quant_converter.npu_quant_grouped_mm_converter(converter, model)

    assert converter.enabled is True
    assert converter.moe_fqns == ["layers.0.moe"]
    assert spy.model is model
    assert spy.config == {"recipe": "mxfp8"}
    assert spy.filter_fn is not None
