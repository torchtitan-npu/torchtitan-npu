# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools
import logging
import os
import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
from .weight_utils import convert_expert_format, detect_expert_format

logger = logging.getLogger(__name__)


@dataclass
class SaveConfig:
    enabled: bool = False
    save_format: str = "dcp"
    save_expert_format: Optional[str] = None
    hf_save_dir: Optional[str] = None
    num_experts: int = 0
    
    _adapter: Optional[Any] = field(default=None, repr=False)
    _patched: bool = field(default=False, repr=False)
    
    def reset(self):
        self.enabled = False
        self.save_format = "dcp"
        self.save_expert_format = None
        self.hf_save_dir = None
        self._adapter = None
        self._patched = False
        self.num_experts = 0

        
_config = SaveConfig()

_original_save: Optional[Callable] = None
_original_model_states_sd: Optional[Callable] = None


def configure_from_model_args(model_args: Any, adapter: Optional[Any] = None):
    def get_config(attr: str, default):
        val = getattr(model_args, attr, None)
        return val if val is not None else default

    _config.enabled = get_config('save_patch_enabled', False)
    _config.save_format = get_config('save_format', 'dcp')
    _config.save_expert_format = get_config('save_expert_format', None)
    _config.hf_save_dir = get_config('hf_save_dir', None)
    _config._adapter = adapter


def is_enabled() -> bool:
    return _config.enabled


def _convert_state_dict_for_save(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """ Convert state_dict to Expert format according to configuration. """
    if not _config.save_expert_format:
        return state_dict

    current = detect_expert_format(state_dict)
    target = _config.save_expert_format
    if _config.save_format == "hf":
        target = "standard"
    if current != target and current != "none":
        logger.info(f"Converting expert format for save: {current} -> {target}")
        return convert_expert_format(state_dict, target)
    
    return state_dict


def _get_total_experts() -> int:
    model_args = getattr(_config._adapter, 'model_args', None)
    return model_args.moe_args.num_experts


def _convert_to_hf_and_save(state_dict: Dict[str, Any], output_dir: str):
    """ Convert to HF format and save. Support EP saving"""

    is_distributed = torch.distributed.is_initialized()
    rank = torch.distributed.get_rank() if is_distributed else 0
    world_size = torch.distributed.get_world_size() if is_distributed else 1
    is_main = (rank == 0)
    
    try:
        from safetensors.torch import save_file
        
        # Filtering non-model parameters
        excluded = ('train_state', 'optimizer', 'lr_scheduler', 'dataloader')
        model_state_dict = {
            k: v for k, v in state_dict.items()
            if not any(k.startswith(p) for p in excluded)
        }
        # Acquire configs and key-mapping
        total_experts = _get_total_experts()
        experts_per_rank = total_experts // world_size if total_experts > 0 and world_size > 1 else 0
        hf_state_dict = _config._adapter.to_hf(model_state_dict)
        
        # Separate expert and non-expert weights
        expert_keys = sorted([k for k in hf_state_dict.keys() if '.experts' in k])
        non_expert_keys = sorted([k for k in hf_state_dict.keys() if '.exeprts.' not in k])
        
        # Determine if remapping is needed
        expert_ids_in_hf = set()
        for key in expert_keys:
            match = re.search(r'\.experts\.(\d+)\.', key)
            if match:
                expert_ids_in_hf.add(int(match.group(1)))
        
        min_expert_id = min(expert_ids_in_hf) if expert_ids_in_hf else 0
        max_expert_id = max(expert_ids_in_hf) if expert_ids_in_hf else 0
        already_global = (max_expert_id >= experts_per_rank) or (rank > 0 and min_expert_id > 0)
        
        if is_main:
            logger.info(f"experts_per_rank = {experts_per_rank}, already_global = {already_global}")
        
        # Handling non-expert weights
        cpu_non_expert = {}
        for key in non_expert_keys:
            value = hf_state_dict[key]
            if hasattr(value, 'full_tensor'):
                full_value = value.full_tensor()
                if is_main:
                    cpu_non_expert[key] = full_value.cpu()
            else:
                if is_main:
                    if isinstance(value, torch.Tensor):
                        cpu_non_expert[key] = value.cpu()
                    else:
                        cpu_non_expert[key] = value
        
        # Handling expert weights
        local_expert_state = {}
        
        for key in expert_keys:
            value = hf_state_dict[key]
            if already_global:
                new_key = key
            else:
                def remap_expert_id(m):
                    local_id = int(m.group(1))
                    global_id = rank * experts_per_rank + local_id
                    return f'.experts.{global_id}.'

                new_key = re.sub(r'\.experts\.(\d+)\.', remap_expert_id, key)
            
            # Convert to CPU tensor
            if hasattr(value, 'full_tensor'):
                full_value = value.full_tensor()
                local_expert_state[new_key] = full_value.cpu()
            elif isinstance(value, torch.Tensor):
                local_expert_state[new_key] = value.cpu()
            else:
                local_expert_state[new_key] = value
        
        # Collect expert weights for all ranks
        if is_distributed and world_size > 1 and experts_per_rank > 0:
            all_expert_states = [None] * world_size
            torch.distributed.all_gather_object(all_expert_states, local_expert_state)
        else:
            all_expert_states = [local_expert_state]
        
        # Rank 0 merge and save
        if is_main:
            merged_experts = {}
            for expert_dict in all_expert_states:
                if expert_dict:
                    merged_experts.update(expert_dict)

            cpu_state = {**cpu_non_expert, **merged_experts}
            
            if not cpu_state:
                logger.warning("No tensors to save")
            else:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "model.safetensors")
                save_file(cpu_state, save_path)
                
                total_size = sum(
                    t.numel() * t.element_size()
                    for t in cpu_state.values()
                    if isinstance(t, torch.Tensor)
                )
                
                # Statistical expert number
                saved_expert_ids = set()
                for key in cpu_state.keys():
                    match = re.search(r'\.experts\.(\d+)\.', key)
                    if match:
                        saved_expert_ids.add(int(match.group(1)))
                
                # generate index.json
                index = {
                    "metadata": {"total_size": total_size},
                    "weight_map": {k: "model.safetensors" for k in sorted(cpu_state.keys())}
                }
                index_path = os.path.join(output_dir, "model.safetensors.index.json")
                with open(index_path, 'w') as f:
                    json.dump(index, f, indent=2)
                sorted_ids = sorted(saved_expert_ids)
                logger.info(
                    f"Saved HF checkpoint to {save_path}"
                    f"({len(cpu_state)} tensors, {len(saved_expert_ids)} experts: {sorted_ids[0]}-{sorted_ids[-1]},"
                    f"{total_size / 1e9:.2f}GB)"
                )
    
    except Exception as e:
        if is_main:
            logger.error(f"Failed to save HF checkpoint: {e}", exc_info=True)
    
    finally:
        if is_distributed:
            torch.distributed.barrier()
                                

def _create_patched_model_states_sd(original_method: Callable) -> Callable:
    """ Wrap _flattened_model_states_sd to convert to expert format"""
    
    @functools.wraps(original_method)
    def patched(self, *args, **kwargs):
        states = original_method(self, *args, **kwargs)
        
        if _config.enabled and _config.save_expert_format:
            return _convert_state_dict_for_save(states)
        return states
    
    return patched


def _create_patched_save(original_save: Callable) -> Callable:
    """ Package the save method to support dual-format saving"""
    
    @functools.wraps(original_save)
    def patched_save(self, curr_step: int, last_step: bool = False):
        if not _config.enabled:
            return original_save(self, curr_step, last_step)
        
        # Check if should save at this step
        should_save = True
        if hasattr(self, 'enable_checkpoint') and not self.enable_checkpoint:
            return None
        if hasattr(self, 'interval'):
            should_save = last_step or (curr_step % self.interval == 0)
        if not should_save:
            return None
        
        result = None
        
        # DCP
        if _config.save_format == "dcp":
            result = original_save(self, curr_step, last_step)
        
        # HF
        if _config.save_format == "hf" and _config.hf_save_dir:
            state_dict = self._flattened_model_states_sd()
            hf_dir = os.path.join(_config.hf_save_dir, f"step_{curr_step}")
            _convert_to_hf_and_save(state_dict, hf_dir)
            
        return result
    
    return patched_save


def apply_patch() -> bool:
    """ apply monkey patch """
    global _original_save, _original_model_states_sd
    
    if _config._patched:
        return True
    
    if not _config.enabled:
        return False
    
    try:
        from torchtitan.components.checkpoint import CheckpointManager
        
        # Patch "_flattened_model_states_sd" expert convertion
        if hasattr(CheckpointManager, '_flattened_model_states_sd'):
            if _original_model_states_sd is None:
                _original_model_states_sd = CheckpointManager._flattened_model_states_sd
            CheckpointManager._flattened_model_states_sd = _create_patched_model_states_sd(_original_model_states_sd)
            
        # Patch "save" file saving
        if _original_save is None:
            _original_save = CheckpointManager.save
        CheckpointManager.save = _create_patched_save(_original_save)
        
        _config._patched = True
        
        return True
    
    except Exception as e:
        logger.error(f"Failed to apply checkpoint patch: {e}", exc_info=True)
        return False