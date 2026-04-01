# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.2.2"

import sys

_initialized = False


def _apply_patches():
    """Apply all patches for torchtitan-npu"""
    global _initialized
    if _initialized:
        return
    _initialized = True
    import torchtitan.models as titan_models

    # patching mxfp8/hif8
    from .converters import quant_converter  # noqa: F401

    # patching models
    from .models import deepseek_v3, llama3  # noqa: F401

    from .models.deepseek_v3.infra import parallelize  # noqa: F401

    # patching context_parallel utils
    from .patches.distributed import context_parallel_utils  # noqa: F401

    # patching optimizer
    from .patches.optimizer import swap_optimizer  # noqa: F401

    # patching step timing
    from .patches.tools import utils  # noqa: F401

    # patching utils
    from .patches.distributed import utils  # noqa: F401, F811

    # async_tp
    # patching torch
    from .patches.torch import clip_grad, micro_pipeline_tp, pipelining  # noqa: F401

    # patching fake process group
    from .patches.torch.testing._internal.distributed import fake_pg  # noqa: F401

    # patching torch_npu
    from .patches.torch_npu import custom_shardings  # noqa: F401

    # patching torchtitan
    from .patches.torchtitan import (  # noqa: F401
        activation_checkpoint,
        hf_datasets,
        lr_scheduler,
    )

    # patching tools
    from .tools import flight_recorder, profiling  # noqa: F401

    new_set = set(titan_models._supported_models)
    new_set.add("deepseek_v32")
    titan_models._supported_models = frozenset(new_set)

    # module injection
    from .models import deepseek_v32  # noqa: F401

    _inject_module("torchtitan.models.deepseek_v32", deepseek_v32)

    # patching model_converter
    from . import converters  # noqa: F401


def _inject_module(module_path: str, replacement_module):
    """add/replace modules into sys.modules"""
    sys.modules[module_path] = replacement_module


_apply_patches()
