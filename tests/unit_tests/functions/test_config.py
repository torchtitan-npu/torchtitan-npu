# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import threading
import types
from unittest.mock import patch

import pytest
import torchtitan.distributed.utils as dist_utils
from torchtitan.train import Trainer

from torchtitan_npu.config.custom_config import (
    JobConfig,
    Optimizer,
    Parallelism,
    Training,
)
from torchtitan_npu.patches.distributed import context_parallel_utils as cp_utils


class DummyContextManager:
    def __init__(self, cp_mesh, buffers, buffer_seq_dims, no_restore_buffers):
        self.cp_mesh = cp_mesh
        self.buffers = buffers
        self.buffer_seq_dims = buffer_seq_dims
        self.no_restore_buffers = no_restore_buffers

    def __enter__(self):
        return self

    @staticmethod
    def __exit__(exc_type, exc, tb):
        return False


class InvalidContextManager:
    pass


class DummyMesh:
    def __init__(self, size_value):
        self.size_value = size_value

    def size(self):
        return self.size_value


def _build_patch_context():
    return threading.local()


def test_optimizer_defaults_expose_swap_config():
    config = Optimizer()

    assert config.swap_optimizer is False
    assert config.swap_optimizer_times == 16
    assert config.name == "AdamW"


def test_parallelism_defaults_expose_custom_context_config():
    config = Parallelism()

    assert config.enable_custom_context_parallel is False
    assert config.custom_context_parallel_path == ""


def test_parallelism_accepts_custom_context_override():
    config = Parallelism(
        enable_custom_context_parallel=True,
        custom_context_parallel_path=(
            "torchtitan_npu.distributed.context_parallel.dsa_cp."
            "AscendDSAContextParallelContext"
        ),
    )

    assert config.enable_custom_context_parallel is True
    assert config.custom_context_parallel_path.endswith(
        "AscendDSAContextParallelContext"
    )


def test_training_defaults_expose_npu_memory_ratio():
    config = Training()

    assert config.torch_npu_memory_ratio == 1.0


def test_job_config_accepts_custom_sections():
    job_config = JobConfig(
        optimizer=Optimizer(swap_optimizer=True, swap_optimizer_times=8),
        parallelism=Parallelism(
            enable_custom_context_parallel=True,
            custom_context_parallel_path="pkg.module.ContextManager",
        ),
        training=Training(torch_npu_memory_ratio=0.8),
    )

    assert job_config.optimizer.swap_optimizer is True
    assert job_config.optimizer.swap_optimizer_times == 8
    assert job_config.parallelism.enable_custom_context_parallel is True
    assert (
        job_config.parallelism.custom_context_parallel_path
        == "pkg.module.ContextManager"
    )
    assert job_config.training.torch_npu_memory_ratio == 0.8


def test_create_cp_ctx_wrapper_uses_original_context_when_custom_disabled():
    sentinel = object()
    mesh = DummyMesh(2)
    parallel_config = Parallelism(enable_custom_context_parallel=False)
    patch_context = _build_patch_context()

    patch_context.current_parallel_config = parallel_config
    patch_context.model_args = types.SimpleNamespace(n_heads=8)

    with patch.object(cp_utils, "_patch_context", patch_context), patch.object(
        cp_utils, "_original_create_cp_ctx", lambda *args: sentinel
    ):
        result = dist_utils.create_context_parallel_ctx(
            mesh, [], [], set(), "allgather"
        )

    assert result is sentinel


def test_create_cp_ctx_wrapper_uses_custom_context_when_enabled(monkeypatch):
    mesh = DummyMesh(2)
    parallel_config = Parallelism(
        enable_custom_context_parallel=True,
        custom_context_parallel_path="pkg.module.CustomContext",
    )
    patch_context = _build_patch_context()

    patch_context.current_parallel_config = parallel_config
    patch_context.model_args = types.SimpleNamespace(n_heads=8)

    monkeypatch.setattr(
        cp_utils, "load_class_from_string", lambda path: DummyContextManager
    )

    with patch.object(cp_utils, "_patch_context", patch_context):
        result = dist_utils.create_context_parallel_ctx(
            mesh, ["buffer"], [1], {"no_restore"}, "allgather"
        )

    assert isinstance(result, DummyContextManager)
    assert result.cp_mesh is mesh
    assert result.buffers == ["buffer"]
    assert result.buffer_seq_dims == [1]
    assert result.no_restore_buffers == {"no_restore"}


def test_create_cp_ctx_wrapper_rejects_non_context_manager(monkeypatch):
    mesh = DummyMesh(2)
    parallel_config = Parallelism(
        enable_custom_context_parallel=True,
        custom_context_parallel_path="pkg.module.InvalidContext",
    )
    patch_context = _build_patch_context()

    patch_context.current_parallel_config = parallel_config
    patch_context.model_args = types.SimpleNamespace(n_heads=8)

    monkeypatch.setattr(
        cp_utils, "load_class_from_string", lambda path: InvalidContextManager
    )

    with patch.object(cp_utils, "_patch_context", patch_context), pytest.raises(
        TypeError, match="is not a context manager"
    ):
        dist_utils.create_context_parallel_ctx(mesh, [], [], set(), "allgather")


def test_create_cp_ctx_wrapper_validates_combined_tp_cp_degree():
    mesh = DummyMesh(2)
    parallel_config = types.SimpleNamespace(tensor_parallel_degree=2)
    patch_context = _build_patch_context()

    patch_context.current_parallel_config = parallel_config
    patch_context.model_args = types.SimpleNamespace(n_heads=7)

    with patch.object(cp_utils, "_patch_context", patch_context), pytest.raises(
        ValueError, match="does not divide the number of heads"
    ):
        dist_utils.create_context_parallel_ctx(mesh, [], [], set(), "allgather")


def test_create_cp_ctx_wrapper_clears_patch_context_after_step():
    patch_context = _build_patch_context()
    patch_context.current_parallel_config = None
    patch_context.model_args = None

    dummy_self = types.SimpleNamespace(
        job_config=types.SimpleNamespace(parallelism="parallel_config"),
        model_args="model_args",
    )

    with patch.object(cp_utils, "_patch_context", patch_context), patch.object(
        cp_utils, "_original_step_method", lambda self, *args, **kwargs: "ok"
    ):
        result = Trainer.forward_backward_step(dummy_self)

    assert result == "ok"
    assert patch_context.current_parallel_config is None
    assert patch_context.model_args is None
