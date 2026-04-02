# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed._tensor import DTensor, Partial, Replicate

from torchtitan_npu.converters.kernels.dsa import SparseLightningIndexerKLLoss
from torchtitan_npu.distributed.context_parallel.dsa_cp import (
    _maybe_to_local_tensor,
    allgather_sequence,
    dsa_forward_with_cp,
    patch_dsa_for_context_parallel,
    ToLocalWithPartialGrad,
)
from torchtitan_npu.models.deepseek_v32.model.model import DSV32_SDPA


def _make_cpu_mesh():
    from torch.distributed.device_mesh import init_device_mesh

    return init_device_mesh("cpu", (1,))


class TestMaybeToLocalTensor:
    @staticmethod
    def test_none_passes_through():
        assert _maybe_to_local_tensor(None) is None

    @staticmethod
    def test_plain_tensor_returns_same_object():
        t = torch.randn(2, 3)
        assert _maybe_to_local_tensor(t) is t

    @staticmethod
    def test_dtensor_calls_to_local():
        local = torch.randn(2, 3)
        mock_dtensor = MagicMock(spec=DTensor)
        mock_dtensor.to_local.return_value = local

        result = _maybe_to_local_tensor(mock_dtensor)

        mock_dtensor.to_local.assert_called_once()
        assert result is local


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

    @pytest.mark.usefixtures("single_rank_process_group")
    @staticmethod
    def test_forward_stores_mesh_and_returns_local():
        mesh = _make_cpu_mesh()
        local = torch.randn(2, 4)
        dtensor = DTensor.from_local(local, mesh, placements=[Replicate()])

        # Apply via autograd to get a real ctx
        result = ToLocalWithPartialGrad.apply(dtensor, mesh)

        assert result.shape == local.shape
        assert torch.allclose(result, local)


class TestAllgatherSequence:
    @pytest.mark.usefixtures("single_rank_process_group")
    @staticmethod
    def test_single_rank_preserves_shape_and_values():
        mesh = _make_cpu_mesh()
        tensor = torch.randn(1, 8, 4, 16)

        result = allgather_sequence(tensor, mesh)

        assert result.shape == tensor.shape
        assert torch.allclose(result, tensor)

    @pytest.mark.usefixtures("single_rank_process_group")
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
