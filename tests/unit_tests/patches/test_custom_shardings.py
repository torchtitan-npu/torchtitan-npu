# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest.mock import MagicMock, patch

import pytest
import torch


class TestModuleImport:
    """Tests for module import and basic functionality."""

    def test_module_imports_successfully(self):
        """Test that the custom_shardings module can be imported."""
        # This test verifies the module structure without requiring NPU
        import importlib.util

        spec = importlib.util.find_spec(
            "torchtitan_npu.patches.torch_npu.custom_shardings"
        )
        assert spec is not None, "Module should be discoverable"


class TestRegisterShardingPatch:
    """Tests for register_sharding_patch function."""

    def test_register_existing_op(self):
        """Test registering a patch for an existing operator."""
        from torch.distributed.tensor import DTensor

        from torchtitan_npu.patches.torch_npu.custom_shardings import (
            register_sharding_patch,
        )

        mock_handler = MagicMock(return_value="original_result")
        mock_patch_fn = MagicMock(return_value="patched_result")

        # Get the real registry
        registry = DTensor._op_dispatcher.sharding_propagator.op_strategy_funcs
        test_op = torch.ops.aten.add.Tensor

        # Save original if exists
        original = registry.get(test_op, None)

        try:
            # Register the op first if not present
            registry[test_op] = mock_handler

            # Register patch
            register_sharding_patch(test_op, mock_patch_fn)

            # Verify the patch was registered
            wrapper = registry[test_op]
            result = wrapper("fake_schema")

            # Patch function should be called with schema and original handler
            mock_patch_fn.assert_called_once_with("fake_schema", mock_handler)
            assert result == "patched_result"

        finally:
            # Restore original state
            if original is not None:
                registry[test_op] = original
            elif test_op in registry:
                del registry[test_op]

    def test_register_nonexistent_op_raises_error(self):
        """Test that registering a patch for non-existent operator raises NotImplementedError."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import (
            register_sharding_patch,
        )

        fake_op = MagicMock()
        fake_op.__repr__ = lambda self: "fake_op"

        with patch(
            "torchtitan_npu.patches.torch_npu.custom_shardings._SHARDING_REGISTRY",
            {},
        ):
            with pytest.raises(
                NotImplementedError, match="not found in original sharding registry"
            ):
                register_sharding_patch(fake_op, MagicMock())


class TestMatmulSharding:
    """Tests for matmul_sharding function."""

    def _create_mock_args_schema(self, shape1, shape2):
        """Helper to create mock args_schema for matmul."""
        mock_spec1 = MagicMock()
        mock_spec1.shape = shape1

        mock_spec2 = MagicMock()
        mock_spec2.shape = shape2

        mock_strategy1 = MagicMock()
        mock_strategy1.output_spec = mock_spec1

        mock_strategy2 = MagicMock()
        mock_strategy2.output_spec = mock_spec2

        mock_arg1 = MagicMock()
        mock_arg1.strategies = [mock_strategy1]

        mock_arg2 = MagicMock()
        mock_arg2.strategies = [mock_strategy2]

        return (mock_arg1, mock_arg2)

    def test_matmul_2d_shapes(self):
        """Test matmul sharding with 2D inputs (mk,kn->mn)."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import matmul_sharding

        mock_schema = MagicMock()
        mock_schema.args_schema = self._create_mock_args_schema((64, 128), (128, 256))
        mock_schema.get_mesh_from_args.return_value = MagicMock()

        mock_fallback = MagicMock(return_value="fallback_result")

        with patch(
            "torchtitan_npu.patches.torch_npu.custom_shardings._mm_like_strategy"
        ) as mock_mm_strategy:
            mock_mm_strategy.return_value = "mm_strategy_result"

            result = matmul_sharding(mock_schema, mock_fallback)

            # Should call _mm_like_strategy with correct equation
            mock_mm_strategy.assert_called_once()
            call_args = mock_mm_strategy.call_args
            assert call_args[0][0] == "mk,kn->mn"
            assert result == "mm_strategy_result"

    def test_matmul_3d_shapes(self):
        """Test matmul sharding with 3D and 2D inputs (bmk,kn->bmn)."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import matmul_sharding

        mock_schema = MagicMock()
        mock_schema.args_schema = self._create_mock_args_schema(
            (4, 64, 128), (128, 256)
        )
        mock_schema.get_mesh_from_args.return_value = MagicMock()

        mock_fallback = MagicMock(return_value="fallback_result")

        with patch(
            "torchtitan_npu.patches.torch_npu.custom_shardings._mm_like_strategy"
        ) as mock_mm_strategy:
            mock_mm_strategy.return_value = "mm_strategy_result"

            result = matmul_sharding(mock_schema, mock_fallback)

            # Should call _mm_like_strategy with correct equation for 3D x 2D
            mock_mm_strategy.assert_called_once()
            call_args = mock_mm_strategy.call_args
            assert call_args[0][0] == "bmk,kn->bmn"
            assert result == "mm_strategy_result"

    def test_matmul_unsupported_num_args(self):
        """Test matmul sharding with unsupported number of arguments."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import matmul_sharding

        mock_schema = MagicMock()
        mock_schema.args_schema = (MagicMock(), MagicMock(), MagicMock())  # 3 args

        with pytest.raises(NotImplementedError, match="Only support two inputs"):
            matmul_sharding(mock_schema, MagicMock())

    def test_matmul_3d_shape_mismatch(self):
        """Test matmul sharding with incompatible 3D shapes."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import matmul_sharding

        # shape1[2] != shape2[0]
        mock_schema = MagicMock()
        mock_schema.args_schema = self._create_mock_args_schema(
            (4, 64, 100), (128, 256)
        )
        mock_schema.get_mesh_from_args.return_value = MagicMock()

        with pytest.raises(
            NotImplementedError, match="Input shapes are not 'bmk' and 'kn'"
        ):
            matmul_sharding(mock_schema, MagicMock())

    def test_matmul_2d_shape_mismatch(self):
        """Test matmul sharding with incompatible 2D shapes."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import matmul_sharding

        # shape1[1] != shape2[0]
        mock_schema = MagicMock()
        mock_schema.args_schema = self._create_mock_args_schema((64, 100), (128, 256))
        mock_schema.get_mesh_from_args.return_value = MagicMock()

        with pytest.raises(
            NotImplementedError, match="Input shapes are not 'mk' and 'kn'"
        ):
            matmul_sharding(mock_schema, MagicMock())

    def test_matmul_fallback_for_unsupported_shapes(self):
        """Test that unsupported shapes fall back to original handler."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import matmul_sharding

        # 4D input - should fallback
        mock_schema = MagicMock()
        mock_schema.args_schema = self._create_mock_args_schema(
            (2, 4, 64, 128), (128, 256)
        )
        mock_schema.get_mesh_from_args.return_value = MagicMock()

        mock_fallback = MagicMock(return_value="fallback_result")

        with patch(
            "torchtitan_npu.patches.torch_npu.custom_shardings._mm_like_strategy"
        ) as mock_mm_strategy:
            result = matmul_sharding(mock_schema, mock_fallback)

            # Should not call _mm_like_strategy
            mock_mm_strategy.assert_not_called()
            # Should call fallback
            mock_fallback.assert_called_once_with(mock_schema)
            assert result == "fallback_result"


class TestCombineStrategiesForMatmulBackward:
    """Tests for combine_strategies_for_matmul_backward function."""

    def test_combine_strategies_success(self):
        """Test successful combination of strategies."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import (
            combine_strategies_for_matmul_backward,
        )

        # Create mock specs with compatible placements
        mock_placement = MagicMock()
        mock_placement.placements = ("Shard", 0)

        # Create dx spec
        dx_input_spec1 = MagicMock()
        dx_input_spec1.placements = mock_placement.placements
        dx_input_spec2 = MagicMock()

        dx_spec = MagicMock()
        dx_spec.input_specs = [dx_input_spec1, dx_input_spec2]
        dx_spec.output_specs = MagicMock()
        dx_spec.redistribute_cost = [[0.1], [0.2]]

        # Create dw spec with matching dy placement
        dw_input_spec0 = MagicMock()
        dw_input_spec1 = MagicMock()
        dw_input_spec1.placements = mock_placement.placements  # Must match dx's dy

        dw_spec = MagicMock()
        dw_spec.input_specs = [dw_input_spec0, dw_input_spec1]
        dw_spec.output_specs = MagicMock()
        dw_spec.redistribute_cost = [[0.3], [0.4]]

        strategy_dx = MagicMock()
        strategy_dx.strategies = [dx_spec]

        strategy_dw = MagicMock()
        strategy_dw.strategies = [dw_spec]

        original_weight_spec = MagicMock()

        with patch(
            "torchtitan_npu.patches.torch_npu.custom_shardings.OpSpec"
        ) as mock_op_spec:
            mock_op_spec.return_value = "combined_spec"

            with patch(
                "torchtitan_npu.patches.torch_npu.custom_shardings.OpStrategy"
            ) as mock_op_strategy:
                mock_op_strategy.return_value = "final_strategy"

                result = combine_strategies_for_matmul_backward(
                    strategy_dx, strategy_dw, original_weight_spec
                )

                # Should create OpSpec and OpStrategy
                mock_op_spec.assert_called_once()
                mock_op_strategy.assert_called_once()
                assert result == "final_strategy"

    def test_combine_strategies_no_compatible_pair(self):
        """Test that RuntimeError is raised when no compatible pairs found."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import (
            combine_strategies_for_matmul_backward,
        )

        # Create specs with incompatible placements
        dx_input_spec1 = MagicMock()
        dx_input_spec1.placements = ("Shard", 0)

        dx_spec = MagicMock()
        dx_spec.input_specs = [dx_input_spec1, MagicMock()]
        dx_spec.output_specs = MagicMock()
        dx_spec.redistribute_cost = [[0.1], [0.2]]

        dw_input_spec1 = MagicMock()
        dw_input_spec1.placements = ("Shard", 1)  # Different placement!

        dw_spec = MagicMock()
        dw_spec.input_specs = [MagicMock(), dw_input_spec1]
        dw_spec.output_specs = MagicMock()
        dw_spec.redistribute_cost = [[0.3], [0.4]]

        strategy_dx = MagicMock()
        strategy_dx.strategies = [dx_spec]

        strategy_dw = MagicMock()
        strategy_dw.strategies = [dw_spec]

        with pytest.raises(
            RuntimeError, match="No compatible matmul_backward strategy"
        ):
            combine_strategies_for_matmul_backward(
                strategy_dx, strategy_dw, MagicMock()
            )


class TestMatmulBackwardSharding:
    """Tests for matmul_backward_sharding function."""

    def _create_backward_args_schema(self, shape_dy, shape_x, shape_w):
        """Helper to create mock args_schema for matmul_backward."""
        shapes = [shape_dy, shape_x, shape_w, None]  # 4 args (last is grad_output)

        args = []
        for shape in shapes:
            if shape is None:
                mock_arg = MagicMock()
                mock_arg.strategies = []
                args.append(mock_arg)
                continue

            mock_spec = MagicMock()
            mock_spec.shape = shape

            mock_strategy = MagicMock()
            mock_strategy.output_spec = mock_spec

            mock_arg = MagicMock()
            mock_arg.strategies = [mock_strategy]
            args.append(mock_arg)

        return tuple(args)

    def test_backward_3d_shapes(self):
        """Test matmul_backward with 3D inputs."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import (
            matmul_backward_sharding,
        )

        # 3D case: dy (bmk), x (bmn), w (nk)
        mock_schema = MagicMock()
        mock_schema.args_schema = self._create_backward_args_schema(
            (4, 8, 128),  # dy: bmk
            (4, 8, 64),  # x: bmn
            (64, 128),  # w: nk
        )
        mock_schema.get_mesh_from_args.return_value = MagicMock()

        mock_fallback = MagicMock(return_value="fallback_result")

        with patch(
            "torchtitan_npu.patches.torch_npu.custom_shardings._mm_like_strategy"
        ) as mock_mm_strategy, patch(
            "torchtitan_npu.patches.torch_npu.custom_shardings.combine_strategies_for_matmul_backward"
        ) as mock_combine:

            mock_mm_strategy.return_value = MagicMock()
            mock_combine.return_value = "combined_result"

            result = matmul_backward_sharding(mock_schema, mock_fallback)

            # Should call _mm_like_strategy twice (for dx and dw)
            assert mock_mm_strategy.call_count == 2
            # Should call combine_strategies_for_matmul_backward
            mock_combine.assert_called_once()
            assert result == "combined_result"

    def test_backward_2d_shapes(self):
        """Test matmul_backward with 2D inputs."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import (
            matmul_backward_sharding,
        )

        mock_schema = MagicMock()
        mock_schema.args_schema = self._create_backward_args_schema(
            (64, 128),  # dy: mk
            (64, 64),  # x: mn
            (64, 128),  # w: nk
        )
        mock_schema.get_mesh_from_args.return_value = MagicMock()

        mock_fallback = MagicMock(return_value="fallback_result")

        with patch(
            "torchtitan_npu.patches.torch_npu.custom_shardings._mm_like_strategy"
        ) as mock_mm_strategy, patch(
            "torchtitan_npu.patches.torch_npu.custom_shardings.combine_strategies_for_matmul_backward"
        ) as mock_combine:

            mock_mm_strategy.return_value = MagicMock()
            mock_combine.return_value = "combined_result"

            result = matmul_backward_sharding(mock_schema, mock_fallback)

            assert mock_mm_strategy.call_count == 2
            mock_combine.assert_called_once()
            assert result == "combined_result"

    def test_backward_unsupported_num_args(self):
        """Test matmul_backward with wrong number of arguments."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import (
            matmul_backward_sharding,
        )

        mock_schema = MagicMock()
        mock_schema.args_schema = (MagicMock(), MagicMock())  # Only 2 args

        with pytest.raises(NotImplementedError, match="Only support four inputs"):
            matmul_backward_sharding(mock_schema, MagicMock())

    def test_backward_3d_shape_mismatch_dy_w(self):
        """Test matmul_backward with shape mismatch between dy and w."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import (
            matmul_backward_sharding,
        )

        # dy[2] != w[1]
        mock_schema = MagicMock()
        mock_schema.args_schema = self._create_backward_args_schema(
            (4, 8, 100),  # dy: bmk (k=100)
            (4, 8, 64),  # x: bmn
            (128, 64),  # w: nk (k=64)
        )
        mock_schema.get_mesh_from_args.return_value = MagicMock()

        with pytest.raises(
            NotImplementedError, match="Input shapes are not 'bmk'.*'nk'"
        ):
            matmul_backward_sharding(mock_schema, MagicMock())

    def test_backward_fallback(self):
        """Test fallback for unsupported shapes."""
        from torchtitan_npu.patches.torch_npu.custom_shardings import (
            matmul_backward_sharding,
        )

        # 4D input - should fallback
        mock_schema = MagicMock()
        mock_schema.args_schema = self._create_backward_args_schema(
            (2, 4, 8, 128),  # 4D dy
            (4, 8, 64),
            (128, 64),
        )
        mock_schema.get_mesh_from_args.return_value = MagicMock()

        mock_fallback = MagicMock(return_value="fallback_result")

        result = matmul_backward_sharding(mock_schema, mock_fallback)

        mock_fallback.assert_called_once_with(mock_schema)
        assert result == "fallback_result"
