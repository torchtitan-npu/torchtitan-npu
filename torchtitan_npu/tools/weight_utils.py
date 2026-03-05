# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import logging 
from pathlib import Path
from typing import Dict, Tuple
from typing import Any, Dict

import torch
from torch.distributed.tensor import DTensor

logger = logging.getLogger(__name__)


def convert_expert_format(state_dict: Dict, target_format: str) -> Dict:
    current_format = detect_expert_format(state_dict)

    if target_format == 'gmm' and current_format == "standard":
        logger.info(f"Converting expert format for save: {current_format} -> {target_format}")
        return fuse_experts(state_dict)
    elif target_format == "standard" and current_format == "gmm":
        logger.info(f"Converting expert format for save: {current_format} -> {target_format}")
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
    
    pending = {}    
    sorted_keys = sorted(state_dict.keys())
    
    for key in sorted_keys:
        if ".moe.experts.w1" in key:
            
            layer_key = key.replace(".w1", "")
            pending[layer_key] = state_dict.pop(key)
        
        elif ".moe.experts.w3" in key:
            layer_key = key.replace(".w3", "")
            if layer_key in pending:
                w1 = pending.pop(layer_key)
                w3 = state_dict.pop(key)
                
                if isinstance(w1, DTensor):
                    fused = _fuse_w1_w3_dtensor(w1, w3)
                else:
                    fused = _fuse_w1_w3_tensor(w1, w3)
                
                state_dict[layer_key + ".w13"] = fused

    return state_dict


def split_fused_experts(state_dict: Dict) -> Dict:
    """ GMM -> Standard : w13 to w1, w3"""
    result = state_dict
    
    w13_keys = [k for k in state_dict.keys() if ".moe.experts.w13" in k]

    for key in w13_keys:
        value = state_dict[key]
        base_key = key.replace(".w13", "")
        
        if isinstance(value, DTensor):
            w1, w3 = _split_w13_dtensor(value)
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


def _fuse_w1_w3_dtensor(w1: DTensor, w3: DTensor) -> DTensor:
    """ fuse DTensor via cpu """
    if w1.device_mesh != w3.device_mesh:
        raise ValueError(
            f"w1 and w3 must have the same device_mesh. "
            f"Got w1: {w1.device_mesh}, w3: {w3.device_mesh}"
        )
    if w1.placements != w3.placements:
        raise ValueError(
            f"w1 and w3 must have the same placements. "
            f"Got w1: {w1.placements}, w3: {w3.placements}"
        )
    device_mesh = w1.device_mesh
    placements = w1.placements
    
    local_w1 = w1.to_local()
    local_w3 = w3.to_local()
    
    device = local_w1.device
    dtype = local_w1.dtype
    
    # w1, w3 to cpu and fuse
    w1_cpu = local_w1.cpu()
    del w1, local_w1
    gc.collect()
    torch.npu.empty_cache()

    w3_cpu = local_w3.cpu()
    del w3, local_w3
    gc.collect()
    torch.npu.empty_cache()
    
    fused_cpu = torch.cat([w1_cpu, w3_cpu], dim=1)
    del w1_cpu, w3_cpu
    gc.collect()  
    
    # back to npu
    fused_local = fused_cpu.to(device=device, dtype=dtype)
    del fused_cpu
    
    return DTensor.from_local(
        fused_local,
        device_mesh=device_mesh,
        placements=placements
    )


def _fuse_w1_w3_tensor(w1: torch.Tensor, w3: torch.Tensor) -> DTensor:
    """ fuse DTensor """
    device = w1.device
    dtype = w1.dtype
    
    # to cpu
    w1_cpu = w1.cpu()
    del w1
    gc.collect()
    torch.npu.empty_cache()

    w3_cpu = w3.cpu()
    del w3
    gc.collect()
    torch.npu.empty_cache()
    
    # cpu fuse
    fused_cpu = torch.cat([w1_cpu, w3_cpu], dim=1)
    del w1_cpu, w3_cpu
    gc.collect()
    
    return fused_cpu.to(device=device, dtype=dtype)


def _split_w13_dtensor(w13: DTensor) -> Tuple[DTensor, DTensor]:
    """ split DTensor """
    local_tensor = w13.to_local()
    chunks = torch.chunk(local_tensor, 2, dim=1)
    local_w1 = chunks[0].clone()
    local_w3 = chunks[1].clone()
    del chunks, local_tensor
    return (
        DTensor.from_local(local_w1, device_mesh=w13.device_mesh, placement=w13.placements),
        DTensor.from_local(local_w3, device_mesh=w13.device_mesh, placement=w13.placements),
    )
    

def _split_w13_for_mapping(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """ Split w13 into w1 and w3 for HF mapping """
    result = {}
    
    for key, value in state_dict.items():
        if ".moe.experts.w13" in key:
            base_key = key.replace('.w13', '')
            
            # Create placeholders w1 and w3
            # For DTensor, the shape needs to be adjusted.
            if isinstance(value, DTensor):
                
                shape = value.shape
                new_shape = (shape[0], shape[1] // 2, shape[2])
                
                from torch.distributed.tensor import zeros as dt_zeros
                w1 = dt_zeros(new_shape, device_mesh=value.device_mesh, placements=value.placements)
                w3 = dt_zeros(new_shape, device_mesh=value.device_mesh, placements=value.placements)
            else:
                half = value.shape[1] // 2
                w1 = torch.empty(value.shape[0], half, value.shape[2], dtype=value.dtype, device=value.device)
                w3 = torch.empty(value.shape[0], half, value.shape[2], dtype=value.dtype, device=value.device)
                
            result[base_key + '.w1'] = w1
            result[base_key + '.w3'] = w3
        else:
            result[key] = value
    
    return result
    