# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging 
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


def convert_expert_format(state_dict: Dict, target_format: str) -> Dict:
    current_format = detect_expert_format(state_dict)

    if target_format == 'gmm' and current_format == "standard":
        return fuse_experts(state_dict)
    elif target_format == "standard" and current_format == "gmm":
        return split_fused_experts(state_dict)
    return state_dict


def detect_expert_format(state_dict: Dict) -> str:
    ''' Detecting the expert format of state_dict '''
    for key in state_dict.keys():
        if '.experts.' in key or 'experts.' in key:
            if '.gate_up_proj.' in key or '.w13.' in key:
                return "gmm"
            if '.gate_proj.' in key or '.w1' in key:
                return "standard"
    return "none"


def detect_input_format_by_path(path: str) -> str:
    """ Check format of input checkpoint by path """
    path = Path(path)
    dcp_markers = ['.distcp', '.metadata', '__0_0.distcp']
    is_dcp = any((path / marker).exists() for marker in dcp_markers)
    if not is_dcp:
        return "hf"
    return "dcp"


def fuse_experts(state_dict: Dict) -> Dict:
    """ Standard -> GMM: w1 w3 to w13"""
    result = state_dict
    
    pending = {}
    keys_to_delete = []
    
    sorted_keys = sorted(state_dict.keys())
    
    for key in sorted_keys:
        value = state_dict[key]
        if ".moe.experts.w1" in key:
            layer_key = key.replace(".w1", "")
            if layer_key not in pending:
                pending[layer_key] = {}
            pending[layer_key]["w1"] = value
            keys_to_delete.append(key)
        
        elif ".moe.experts.w3" in key:
            layer_key = key.replace(".w3", "")
            if layer_key not in pending:
                pending[layer_key] = {}
            pending[layer_key]["w3"] = value
            keys_to_delete.append(key)
            
            # fuse immediately
            if "w1" in pending.get(layer_key, {}):
                w1 = pending[layer_key]["w1"]
                w3 = pending[layer_key]["w3"]
                
                if isinstance(w1, DTensor):
                    fused = _fuse_dtensor("w1")
                else:
                    fused = torch.cat([w1, w3], dim=1)
                
                # delete keys and free caches
                result[layer_key + ".w13"] = fused
                w1_key = layer_key + ".w1"
                w3_key = layer_key + ".w3"
                if w1_key in state_dict:
                    del state_dict[w1_key]
                if w3_key in state_dict:
                    del state_dict[w3_key]
                if w1_key in keys_to_delete:
                    keys_to_delete.remove(w1_key)
                if w3_key in keys_to_delete:
                    keys_to_delete.remove(w3_key)
                
                del w1, w3
                gc.collect()
                torch.npu.empty_cache()
    return result


def split_fused_experts(state_dict: Dict) -> Dict:
    """ GMM -> Standard : w13 to w1, w3"""
    result = state_dict
    
    w13_keys = [k for k in state_dict.keys() if ".moe.experts.w13" in k]

    for key in w13_keys:
        value = state_dict[key]
        base_key = key.replace(".w13", "")
        
        if isinstance(value, DTensor):
            w1, w3 = _split_dtensor(value)
        else:
            chunks = torch.chunk(value, 2, dim=1)
            w1 = chunks[0].clone()
            w3 = chunks[1].clone()
            del chunks
            
        result[base_key + ".w1"] = w1
        result[base_key + ".w3"] = w3
        del value
        gc.collect()
    
    return result


def _fuse_dtensor(w1: DTensor, w3: DTensor) -> DTensor:
    """ fuse DTensor """
    local_fused = torch.cat([w1.to_local(), w3.to_local()], dim=1)
    return DTensor.from_local(
        local_fused,
        device_mesh=w1.device_mesh,
        placements=w1.placements,
    )


def _split_dtensor(w13: DTensor) -> Tuple[DTensor, DTensor]:
    """ split DTensor """
    local_tensor = w13.to_local()
    chunks = torch.chunk(local_tensor, 2, dim=1)
    local_w1 = chunks[0].clone
    local_w3 = chunks[1].clone
    del chunks, local_tensor
    return (
        DTensor.from_local(local_w1, device_mesh=w13.device_mesh, placement=w13.placements),
        DTensor.from_local(local_w3, device_mesh=w13.device_mesh, placement=w13.placements),
    )
    