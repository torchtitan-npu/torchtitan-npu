# Adapted from
# https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_api.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
import types
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union, Dict, Type
from typing import OrderedDict as OrderedDictType

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
from torchtitan_npu.converter.kernels.quant_linear import MXLinear
from torchtitan_npu.converter.registry import BaseKernel, KernelType, replace_functions
from torchtitan_npu.converter.kernels.quant_gmm import npu_grouped_mxfp8_mm, npu_grouped_hif8_mm
from .quant_config import MXLinearConfig, MoEGroupedRecipeName

logger = logging.getLogger(__name__)

_QUANTIZE_CONFIG_HANDLER: Dict[
    Type[Any],
    Callable[[torch.nn.Module, Any], torch.nn.Module],
] = {}


def register_quantize_module_handler(config_type):
    def decorator(func):
        _QUANTIZE_CONFIG_HANDLER[config_type] = func
        return func
    return decorator


def _is_linear(mod, *args):
    return(
        isinstance(mod, torch.nn.Linear)
        and hasattr(mod, "weight")
        and not isinstance(mod, nn.modules.linear.NonDynamicallyQuantizableLinear)
    )


def _replace_with_custom_fn_if_matches_filter(
        model,
        replacement_fn,
        filter_fn,
        cur_fqn="",
        extra_args: Optional[Tuple[Any, ...]] = (),
) -> None:
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model, *extra_args)
        return model
    else:
        named_children_list = list(model.named_children())
        for name, child in named_children_list:
            new_child = _replace_with_custom_fn_if_matches_filter(
                child,
                replacement_fn,
                filter_fn,
                f"{cur_fqn}{name}.",
                extra_args,
            )
            if new_child is not child and new_child is not None:
                setattr(model, name, new_child)

        return model


def linear_quantize_(
        model: torch.nn.Module,
        config,
        filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = _is_linear
):
    """
    Convert the weight of linear modules in the model with 'config', model is modified inplace
    Args:
        model (torch.nn.Module): input model
        config: a workflow configuration object.
        filter_fn (Optional[Callable[torch.nn.Module, str], bool]): function that takes a nn.Module instance
        and fully qualified name of the module, returns True if  we want to run 'config' on the weight of the module.
    """
    filter_fn = _is_linear if filter_fn is None else filter_fn
    handler = _QUANTIZE_CONFIG_HANDLER[type(config)]
    # for each linear in the model, apply the transform if filtering passes
    _replace_with_custom_fn_if_matches_filter(
        model,
        handler,
        filter_fn,
        extra_args=(config,),
    )


def grouped_quantize_(
        model: torch.nn.Module,
        config,
        filter_fn: Optional[Callable[[torch.nn.Module, str], bool]] = None,
        recipe_name=None
):
    TARGET_PACKAGE = "torchtitan_npu.converter.kernels.gmm"
    replacement_counts = 0
    if config.recipe_name == MoEGroupedRecipeName.GMM_MXFP8:
        func_replacements = replace_functions(
            func_name="npu_grouped_mm",
            new_func=npu_grouped_mxfp8_mm,
            package=TARGET_PACKAGE
        )
        replacement_counts += func_replacements
    elif config.recipe_name == MoEGroupedRecipeName.GMM_HIF8:
        func_replacements = replace_functions(
            func_name="npu_grouped_mm",
            new_func=npu_grouped_hif8_mm,
            package=TARGET_PACKAGE
        )
        replacement_counts += func_replacements
    else:
        raise AssertionError(f"unknown recipe_name {recipe_name}")
    logger.info(f"[MXFP8/Hif8 GMM] Replaced {replacement_counts} NPU GMM methods/functions.")


@register_quantize_module_handler(MXLinearConfig)
def _mx_linear_transform(module: torch.nn.Module, config: MXLinearConfig):
    return MXLinear.from_float(module, config=config)