# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from torchtitan_npu.config.custom_config import (
    JobConfig,
    Optimizer,
    Parallelism,
    Profiling,
    Training,
)


def test_optimizer_defaults_expose_swap_config():
    config = Optimizer()

    assert config.swap_optimizer is False
    assert config.swap_optimizer_times == 16
    assert config.name == "AdamW"


def test_parallelism_defaults_expose_custom_context_config():
    config = Parallelism()

    assert config.enable_custom_context_parallel is False


def test_parallelism_accepts_custom_context_override():
    config = Parallelism(enable_custom_context_parallel=True)

    assert config.enable_custom_context_parallel is True


def test_training_defaults_expose_npu_memory_ratio():
    config = Training()

    assert config.torch_npu_memory_ratio == 1.0


def test_job_config_accepts_custom_sections():
    job_config = JobConfig(
        optimizer=Optimizer(swap_optimizer=True, swap_optimizer_times=8),
        parallelism=Parallelism(enable_custom_context_parallel=True),
        training=Training(torch_npu_memory_ratio=0.8),
    )

    assert job_config.optimizer.swap_optimizer is True
    assert job_config.optimizer.swap_optimizer_times == 8
    assert job_config.parallelism.enable_custom_context_parallel is True
    assert job_config.training.torch_npu_memory_ratio == 0.8


def test_profiling_defaults_expose_custom_profile_fields():
    config = Profiling()

    assert config.profile_step_start == 0
    assert config.profile_step_end == 0
    assert config.profile_ranks == [-1]
    assert config.profile_record_shapes is True
    assert config.profile_with_memory is False
    assert config.profile_with_stack is False
    assert config.enable_online_parse is True


def test_job_config_uses_custom_config_types_by_default():
    job_config = JobConfig()

    assert isinstance(job_config.optimizer, Optimizer)
    assert isinstance(job_config.parallelism, Parallelism)
    assert isinstance(job_config.training, Training)
    assert isinstance(job_config.profiling, Profiling)
