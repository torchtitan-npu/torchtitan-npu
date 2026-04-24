# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import patch

from torchtitan_npu.converters.registry import ConverterRegistry, PatchInfo, registry


class DummyPatch:
    SUPPORTED_MODELS = {"dummy_model"}


def _run_register_case(register_name, *, supported_models=None):
    calls = []
    test_registry = ConverterRegistry()
    with patch.object(test_registry, "_patches", {}), patch.object(
        test_registry,
        "_register_as_model_converter",
        lambda name, patch_cls, registered_models: calls.append(
            (name, patch_cls, registered_models)
        ),
    ):
        if supported_models is None:
            decorated_cls = test_registry.register(register_name)(DummyPatch)
        else:
            decorated_cls = test_registry.register(
                register_name,
                supported_models=supported_models,
            )(DummyPatch)
        patch_info = test_registry.get(register_name)

    return decorated_cls, patch_info, calls


def test_registry_is_singleton():
    registry1 = ConverterRegistry()
    registry2 = ConverterRegistry()

    assert registry1 is registry2
    assert registry1 is registry


def test_patch_info_is_dataclass_with_expected_defaults():
    patch_info = PatchInfo(name="dummy", patch_cls=DummyPatch)

    assert patch_info.name == "dummy"
    assert patch_info.patch_cls is DummyPatch
    assert patch_info.supported_models == {"*"}


def test_register_uses_patch_supported_models_by_default():
    decorated_cls, _, calls = _run_register_case("unit_dummy")

    assert decorated_cls is DummyPatch
    assert calls == [("unit_dummy", DummyPatch, {"dummy_model"})]


def test_register_supports_explicit_supported_models_override():
    decorated_cls, patch_info, calls = _run_register_case(
        "unit_override",
        supported_models={"model_a", "model_b"},
    )

    assert decorated_cls is DummyPatch
    assert patch_info is not None
    assert patch_info.supported_models == {"model_a", "model_b"}
    assert calls == [("unit_override", DummyPatch, {"model_a", "model_b"})]


def test_get_returns_none_for_unknown_patch():
    assert registry.get("definitely_missing_patch") is None


def test_core_converter_registrations_exist():
    for name in ["npu_dsa", "npu_rms_norm", "npu_rope", "npu_gmm"]:
        patch_info = registry.get(name)
        assert patch_info is not None, f"{name} should be registered"
        assert patch_info.name == name
        assert patch_info.patch_cls is not None
