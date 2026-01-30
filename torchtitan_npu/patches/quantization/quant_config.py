# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union
import torch


class MXLinearRecipeName(Enum):
    FLOAT8_MXFP8 = "mxfp8"
    FLOAT8_HIF8 = "hif8"


class MoEGroupedRecipeName(Enum):
    GMM_MXFP8 = "mxfp8"
    GMM_HIF8 = "hif8"


@dataclass
class MXLinearConfig:
    # element dtype, used for activations, weights and gradients
    # Currently, both forward and backward passes in MXFP8 training use E4M3 data format.
    elem_dtype: Any = torch.float8_e4m3fn
    recipe_name: MXLinearRecipeName = MXLinearRecipeName.FLOAT8_MXFP8

    @staticmethod
    def from_recipe_name(
        recipe_name: Union[MXLinearRecipeName, str],
    ) -> "MXLinearConfig":
        """
        Input: 'MXLinearRecipeName' value, or a string representing a 'MXLinearRecipeName' value
        Output: a 'MXLinearConfig' configured to implement the specified recipe
        """
        if isinstance(recipe_name, str):
            valid_names = [n.value for n in MXLinearRecipeName]
            if recipe_name not in valid_names:
                raise ValueError(f"recipe_name {recipe_name} not in valid names {valid_names}")
            recipe_name = MXLinearRecipeName(recipe_name)
        
        # Return the corresponding quantization data format
        if recipe_name is MXLinearRecipeName.FLOAT8_MXFP8:
            return MXLinearConfig(recipe_name=MXLinearRecipeName.FLOAT8_MXFP8)
        elif recipe_name is MXLinearRecipeName.FLOAT8_HIF8:
            return MXLinearConfig(recipe_name=MXLinearRecipeName.FLOAT8_HIF8)
        else:
            raise AssertionError(f"unknown recipe_name {recipe_name}")


@dataclass
class MoETrainingConfig:
    # element dtype, used for activations, weights and gradients
    elem_dtype: Any = torch.float8_e4m3fn
    recipe_name: MoEGroupedRecipeName = MoEGroupedRecipeName.GMM_MXFP8

    @staticmethod
    def from_recipe_name(
        recipe_name: Union[MoEGroupedRecipeName, str],
    ) -> "MoETrainingConfig":
        """
        Input: 'MoEGroupedRecipeName' value, or a string representing a 'MoEGroupedRecipeName' value
        Output: a 'MoEGroupedRecipeName' configured to implement the specified recipe
        """
        if isinstance(recipe_name, str):
            valid_names = [n.value for n in MoEGroupedRecipeName]
            if recipe_name not in valid_names:
                raise ValueError(f"recipe_name {recipe_name} not in valid names {valid_names}")
            recipe_name = MoEGroupedRecipeName(recipe_name)
        
        # Return the corresponding quantization data format
        if recipe_name is MoEGroupedRecipeName.GMM_MXFP8:
            return MoETrainingConfig(recipe_name=MoEGroupedRecipeName.GMM_MXFP8)
        elif recipe_name is MoEGroupedRecipeName.GMM_HIF8:
            return MoETrainingConfig(recipe_name=MoEGroupedRecipeName.GMM_HIF8)
        else:
            raise AssertionError(f"unknown recipe_name {recipe_name}")