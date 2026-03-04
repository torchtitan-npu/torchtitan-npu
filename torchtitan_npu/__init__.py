# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

__version__ = "0.2.1"

import sys

_initialized = False


def _apply_patches():
    """ Apply all patches for torchtitan-npu"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    # patching tools
    from .tools import profiling
    from .tools import flight_recorder
    # patching step timing
    from .patches.tools import utils
    # patching torch
    from .patches.torch import pipelining
    # patching torchtitan
    from .patches.torchtitan import hf_datasets
    # patching optimizer
    from .patches.optimizer import swap_optimizer
    # patching context_parallel utils
    from .patches.distributed import context_parallel_utils
    # patching torch_npu
    from .patches.torch_npu import custom_shardings

    # patching models
    from .models import deepseek_v3, llama3

    # patching mxfp8/hif8
    from .converters import quant_converter

    import torchtitan.models as titan_models
    new_set = set(titan_models._supported_models)
    new_set.add("deepseek_v32")
    titan_models._supported_models = frozenset(new_set)

    # module injection
    from .models import deepseek_v32
    _inject_module("torchtitan.models.deepseek_v32", deepseek_v32)

    # patching model_converter
    from . import converters


def _inject_module(module_path: str, replacement_module):
    """ add/replace modules into sys.modules"""
    sys.modules[module_path] = replacement_module


_apply_patches()