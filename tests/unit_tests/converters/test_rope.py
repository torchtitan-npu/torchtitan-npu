# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


def test_deepseek_v3_mapping():
    from torchtitan_npu.converters.kernels.rope import RoPEKernel

    impl = RoPEKernel.get_impl_cls("deepseek_v3")

    assert impl is not None
    assert (
        "deepseek" in impl.__name__.lower()
        or impl.__name__ == "npu_apply_rotary_emb_deepseek"
    )


def test_deepseek_v32_mapping():
    from torchtitan_npu.converters.kernels.rope import RoPEKernel

    impl = RoPEKernel.get_impl_cls("deepseek_v32")

    assert impl is not None
    assert (
        "deepseek" in impl.__name__.lower()
        or impl.__name__ == "npu_apply_rotary_emb_deepseek"
    )


def test_qwen3_mapping():
    from torchtitan_npu.converters.kernels.rope import RoPEKernel

    impl = RoPEKernel.get_impl_cls("qwen3")

    assert impl is not None
    assert (
        "qwen" in impl.__name__.lower() or impl.__name__ == "npu_apply_rotary_emb_qwen"
    )


def test_llama_mapping():
    from torchtitan_npu.converters.kernels.rope import (
        npu_apply_rotary_emb_llama,
        RoPEKernel,
    )

    impl = RoPEKernel.get_impl_cls("llama3")
    default_impl = RoPEKernel.MODEL_IMPL.get("_default")

    assert default_impl == npu_apply_rotary_emb_llama
    assert impl == default_impl


def test_unknown_mapping():
    from torchtitan_npu.converters.kernels.rope import (
        npu_apply_rotary_emb_llama,
        RoPEKernel,
    )

    impl = RoPEKernel.get_impl_cls("unknown")
    default_impl = RoPEKernel.MODEL_IMPL.get("_default")

    assert default_impl == npu_apply_rotary_emb_llama
    assert impl == default_impl
