# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed._tensor import DTensor, Partial, Replicate
from torchtitan.models.attention import ScaledDotProductAttentionWrapper

from torchtitan_npu.converters.kernels.dsa import SparseLightningIndexerKLLoss
from torchtitan_npu.distributed.context_parallel.dsa_cp import (
    allgather_sequence,
    dsa_forward_with_cp,
    patch_dsa_for_context_parallel,
    ToLocalWithPartialGrad,
)
from torchtitan_npu.distributed.context_parallel.ulysses_cp import (
    all_to_all,
    AllToAll,
    patch_ulysses_for_context_parallel,
)
from torchtitan_npu.models.deepseek_v32.model.model import DSV32_SDPA
from torchtitan_npu.patches.distributed.custom_context_parallel import (
    validate_ulysses_configs,
)


def _make_cpu_mesh():
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (1,))


@pytest.mark.usefixtures("single_rank_process_group")
class TestToLocalWithPartialGrad:
    @staticmethod
    def test_backward_returns_none_none_for_none_grad():
        ctx = MagicMock()
        result = ToLocalWithPartialGrad.backward(ctx, None)
        assert result == (None, None)

    @staticmethod
    def test_backward_wraps_grad_in_partial_placement():
        mock_mesh = MagicMock()
        mock_ctx = MagicMock()
        mock_ctx.mesh = mock_mesh

        grad_output = torch.randn(2, 4)
        mock_partial_dtensor = MagicMock()

        with patch(
            "torchtitan_npu.distributed.context_parallel.dsa_cp.DTensor"
        ) as mock_dtensor_cls:
            mock_dtensor_cls.from_local.return_value = mock_partial_dtensor

            grad_dtensor, grad_mesh = ToLocalWithPartialGrad.backward(
                mock_ctx, grad_output
            )

            mock_dtensor_cls.from_local.assert_called_once()
            call_args = mock_dtensor_cls.from_local.call_args
            assert call_args[0][0] is grad_output
            assert call_args[0][1] is mock_mesh
            placements = (
                call_args.kwargs["placements"]
                if "placements" in call_args.kwargs
                else call_args[0][2]
            )
            assert len(placements) == 1
            assert isinstance(placements[0], Partial)

            assert grad_dtensor is mock_partial_dtensor
            assert grad_mesh is None

    @staticmethod
    def test_forward_stores_mesh_and_returns_local():
        mesh = _make_cpu_mesh()
        local = torch.randn(2, 4)
        dtensor = DTensor.from_local(local, mesh, placements=[Replicate()])

        # Apply via autograd to get a real ctx
        result = ToLocalWithPartialGrad.apply(dtensor, mesh)

        assert result.shape == local.shape
        assert torch.allclose(result, local)


@pytest.mark.usefixtures("single_rank_process_group")
class TestAllgatherSequence:
    @staticmethod
    def test_single_rank_preserves_shape_and_values():
        mesh = _make_cpu_mesh()
        tensor = torch.randn(1, 8, 4, 16)

        result = allgather_sequence(tensor, mesh)

        assert result.shape == tensor.shape
        assert torch.allclose(result, tensor)

    @staticmethod
    def test_output_is_plain_tensor():
        mesh = _make_cpu_mesh()
        tensor = torch.randn(1, 4, 2, 8)

        result = allgather_sequence(tensor, mesh)

        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, DTensor)


class TestPatchDsaForContextParallel:
    @staticmethod
    def test_sets_cp_mesh_on_class():
        snapshot = TestPatchDsaForContextParallel._snapshot_class_state()
        try:
            mock_mesh = MagicMock()
            patch_dsa_for_context_parallel(cp_mesh=mock_mesh)
            assert DSV32_SDPA.cp_mesh is mock_mesh
        finally:
            TestPatchDsaForContextParallel._restore_class_state(snapshot)

    @staticmethod
    def test_sets_model_args_when_provided():
        snapshot = TestPatchDsaForContextParallel._snapshot_class_state()
        try:
            mock_mesh = MagicMock()
            mock_args = MagicMock()
            patch_dsa_for_context_parallel(cp_mesh=mock_mesh, model_args=mock_args)
            assert DSV32_SDPA.model_args is mock_args
        finally:
            TestPatchDsaForContextParallel._restore_class_state(snapshot)

    @staticmethod
    def test_does_not_set_model_args_when_none():
        snapshot = TestPatchDsaForContextParallel._snapshot_class_state()
        try:
            # Remove any pre-existing model_args
            if hasattr(DSV32_SDPA, "model_args"):
                delattr(DSV32_SDPA, "model_args")

            mock_mesh = MagicMock()
            patch_dsa_for_context_parallel(cp_mesh=mock_mesh, model_args=None)
            assert not hasattr(DSV32_SDPA, "model_args")
        finally:
            TestPatchDsaForContextParallel._restore_class_state(snapshot)

    @staticmethod
    def test_forward_replaced_with_wrapper():
        snapshot = TestPatchDsaForContextParallel._snapshot_class_state()
        try:
            original = DSV32_SDPA.forward
            mock_mesh = MagicMock()
            patch_dsa_for_context_parallel(cp_mesh=mock_mesh)
            assert DSV32_SDPA.forward is not original
        finally:
            TestPatchDsaForContextParallel._restore_class_state(snapshot)

    @staticmethod
    def test_compute_dsa_indexer_loss_is_sparse_lightning_kl_loss():
        snapshot = TestPatchDsaForContextParallel._snapshot_class_state()
        try:
            mock_mesh = MagicMock()
            patch_dsa_for_context_parallel(cp_mesh=mock_mesh)
            assert isinstance(
                DSV32_SDPA.compute_dsa_indexer_loss, SparseLightningIndexerKLLoss
            )
        finally:
            TestPatchDsaForContextParallel._restore_class_state(snapshot)

    @staticmethod
    def test_forward_wrapper_wraps_dsa_forward_with_cp():
        snapshot = TestPatchDsaForContextParallel._snapshot_class_state()
        try:
            mock_mesh = MagicMock()
            patch_dsa_for_context_parallel(cp_mesh=mock_mesh)
            assert DSV32_SDPA.forward is dsa_forward_with_cp
        finally:
            TestPatchDsaForContextParallel._restore_class_state(snapshot)

    @staticmethod
    def _snapshot_class_state():
        return {
            "forward": DSV32_SDPA.forward,
            "had_cp_mesh": hasattr(DSV32_SDPA, "cp_mesh"),
            "had_model_args": hasattr(DSV32_SDPA, "model_args"),
            "had_loss": hasattr(DSV32_SDPA, "compute_dsa_indexer_loss"),
        }

    @staticmethod
    def _restore_class_state(snapshot):
        DSV32_SDPA.forward = snapshot["forward"]
        for attr, had in [
            ("cp_mesh", snapshot["had_cp_mesh"]),
            ("model_args", snapshot["had_model_args"]),
            ("compute_dsa_indexer_loss", snapshot["had_loss"]),
        ]:
            if not had and hasattr(DSV32_SDPA, attr):
                delattr(DSV32_SDPA, attr)


class TestDsaForwardWithCpValidation:
    @staticmethod
    def test_raises_if_k_has_more_than_one_kv_head():
        self_mock = TestDsaForwardWithCpValidation._make_self()
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 2, 1, 16)
        v = torch.randn(1, 1, 1, 16)

        with pytest.raises(NotImplementedError, match="num_head_kv == 1"):
            dsa_forward_with_cp(self_mock, q, k, v)

    @staticmethod
    def test_raises_if_v_has_more_than_one_kv_head():
        self_mock = TestDsaForwardWithCpValidation._make_self()
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 1, 16)
        v = torch.randn(1, 3, 1, 16)

        with pytest.raises(NotImplementedError, match="num_head_kv == 1"):
            dsa_forward_with_cp(self_mock, q, k, v)

    @staticmethod
    def test_raises_only_when_shape_is_wrong():
        self_mock = TestDsaForwardWithCpValidation._make_self()
        q = torch.randn(1, 1, 4, 16)
        k = torch.randn(1, 1, 1, 16)
        v = torch.randn(1, 1, 1, 16)

        # Expect to pass the shape-check and fail later (on allgather / NPU calls),
        # i.e. NOT a NotImplementedError from the shape guard.
        with pytest.raises(Exception) as exc_info:
            dsa_forward_with_cp(self_mock, q, k, v)

        assert "num_head_kv == 1" not in str(exc_info.value)

    @staticmethod
    def _make_self():
        self_mock = MagicMock()
        self_mock.cp_mesh = MagicMock()
        return self_mock


# ---------------------------------------------------------------------------
# Ulysses CP tests
# ---------------------------------------------------------------------------


def _make_cpu_mesh_ulysses():
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (1,))


@pytest.fixture
def mock_all_to_all_identity_for_gloo():
    def _fake_all_to_all(
        output_tensor_list, input_tensor_list, group=None, async_op=False
    ):
        assert len(output_tensor_list) == len(input_tensor_list)
        for out_t, in_t in zip(output_tensor_list, input_tensor_list):
            out_t.copy_(in_t)
        return None

    with patch.object(torch.distributed, "all_to_all", side_effect=_fake_all_to_all):
        yield


@pytest.mark.usefixtures(
    "single_rank_process_group", "mock_all_to_all_identity_for_gloo"
)
class TestAllToAll:
    @staticmethod
    def test_single_rank_preserves_shape_and_values():
        mesh = _make_cpu_mesh_ulysses()
        t = torch.randn(2, 4, 8, 16)

        result = all_to_all(t, mesh, scatter_dim=1, gather_dim=2)

        assert result.shape == t.shape
        assert torch.allclose(result, t)

    @staticmethod
    def test_output_is_plain_tensor():
        mesh = _make_cpu_mesh_ulysses()
        t = torch.randn(1, 4, 4, 8)

        result = all_to_all(t, mesh, scatter_dim=1, gather_dim=2)

        assert isinstance(result, torch.Tensor)
        assert not isinstance(result, DTensor)

    @staticmethod
    def test_backward_grad_shape_matches_input():
        mesh = _make_cpu_mesh_ulysses()
        t = torch.randn(2, 4, 8, 16, requires_grad=True)

        output = AllToAll.apply(t, mesh, 1, 2)
        output.sum().backward()

        assert t.grad is not None
        assert t.grad.shape == t.shape

    @staticmethod
    def test_single_rank_backward_is_identity():
        mesh = _make_cpu_mesh_ulysses()
        t = torch.randn(2, 4, 8, 16, requires_grad=True)
        grad_ref = torch.ones(2, 4, 8, 16)

        output = AllToAll.apply(t, mesh, 1, 2)
        output.backward(torch.ones_like(output))

        assert torch.allclose(t.grad, grad_ref)


@pytest.mark.usefixtures("single_rank_process_group")
class TestPatchUlyssesForContextParallel:
    @staticmethod
    def test_sets_cp_mesh_on_class():
        snapshot = TestPatchUlyssesForContextParallel._snapshot_class_state()
        try:
            mock_mesh = MagicMock()
            patch_ulysses_for_context_parallel(cp_mesh=mock_mesh)
            assert ScaledDotProductAttentionWrapper.cp_mesh is mock_mesh
        finally:
            TestPatchUlyssesForContextParallel._restore_class_state(snapshot)

    @staticmethod
    def test_forward_replaced_with_wrapper():
        snapshot = TestPatchUlyssesForContextParallel._snapshot_class_state()
        try:
            original = ScaledDotProductAttentionWrapper.forward
            mock_mesh = MagicMock()
            patch_ulysses_for_context_parallel(cp_mesh=mock_mesh)
            assert ScaledDotProductAttentionWrapper.forward is not original
        finally:
            TestPatchUlyssesForContextParallel._restore_class_state(snapshot)

    @staticmethod
    def test_patched_forward_calls_all_to_all_for_qkv_and_output():
        snapshot = TestPatchUlyssesForContextParallel._snapshot_class_state()
        try:
            mesh = _make_cpu_mesh_ulysses()
            patch_ulysses_for_context_parallel(cp_mesh=mesh)

            call_count = {"n": 0}

            def counting_a2a(tensor, m, scatter_dim, gather_dim):
                call_count["n"] += 1
                return tensor

            module = ScaledDotProductAttentionWrapper()
            q = torch.randn(1, 4, 8, 16)
            k = torch.randn(1, 4, 8, 16)
            v = torch.randn(1, 4, 8, 16)

            with patch(
                "torchtitan_npu.distributed.context_parallel.ulysses_cp.all_to_all",
                side_effect=counting_a2a,
            ):
                ScaledDotProductAttentionWrapper.forward(module, q, k, v)

            assert call_count["n"] == 4
        finally:
            TestPatchUlyssesForContextParallel._restore_class_state(snapshot)

    @staticmethod
    def _snapshot_class_state():
        return {
            "forward": ScaledDotProductAttentionWrapper.forward,
            "had_cp_mesh": hasattr(ScaledDotProductAttentionWrapper, "cp_mesh"),
        }

    @staticmethod
    def _restore_class_state(snapshot):
        ScaledDotProductAttentionWrapper.forward = snapshot["forward"]
        if not snapshot["had_cp_mesh"] and hasattr(
            ScaledDotProductAttentionWrapper, "cp_mesh"
        ):
            delattr(ScaledDotProductAttentionWrapper, "cp_mesh")


class TestValidateUlyssesConfigs:
    @staticmethod
    def test_passes_for_valid_config():
        validate_ulysses_configs(
            job_config=TestValidateUlyssesConfigs._make_job_config(),
            model_args=TestValidateUlyssesConfigs._make_model_args(n_heads=128),
            cp_mesh=TestValidateUlyssesConfigs._make_cp_mesh(2),
        )

    @staticmethod
    def test_raises_when_n_heads_not_divisible_by_cp_degree():
        with pytest.raises(ValueError, match="n_heads"):
            validate_ulysses_configs(
                job_config=TestValidateUlyssesConfigs._make_job_config(),
                model_args=TestValidateUlyssesConfigs._make_model_args(n_heads=128),
                cp_mesh=TestValidateUlyssesConfigs._make_cp_mesh(3),
            )

    @staticmethod
    def test_raises_when_seq_len_not_divisible_by_cp_degree():
        with pytest.raises(ValueError, match="seq_len"):
            validate_ulysses_configs(
                job_config=TestValidateUlyssesConfigs._make_job_config(seq_len=2049),
                model_args=TestValidateUlyssesConfigs._make_model_args(),
                cp_mesh=TestValidateUlyssesConfigs._make_cp_mesh(2),
            )

    @staticmethod
    def test_raises_when_n_heads_not_divisible_by_tp_times_cp():
        with pytest.raises(ValueError, match="tp_degree"):
            validate_ulysses_configs(
                job_config=TestValidateUlyssesConfigs._make_job_config(tp_degree=3),
                model_args=TestValidateUlyssesConfigs._make_model_args(n_heads=128),
                cp_mesh=TestValidateUlyssesConfigs._make_cp_mesh(2),
            )

    @staticmethod
    def test_passes_without_model_args():
        validate_ulysses_configs(
            job_config=TestValidateUlyssesConfigs._make_job_config(),
            model_args=None,
            cp_mesh=TestValidateUlyssesConfigs._make_cp_mesh(2),
        )

    @staticmethod
    def test_passes_without_job_config():
        validate_ulysses_configs(
            job_config=None,
            model_args=TestValidateUlyssesConfigs._make_model_args(n_heads=128),
            cp_mesh=TestValidateUlyssesConfigs._make_cp_mesh(2),
        )

    @staticmethod
    def _make_cp_mesh(cp_degree):
        mesh = MagicMock()
        mesh.size.return_value = cp_degree
        return mesh

    @staticmethod
    def _make_job_config(seq_len=2048, tp_degree=1):
        job_config = MagicMock()
        job_config.training.seq_len = seq_len
        job_config.parallelism.tensor_parallel_degree = tp_degree
        return job_config

    @staticmethod
    def _make_model_args(n_heads=128):
        model_args = MagicMock()
        model_args.n_heads = n_heads
        return model_args
