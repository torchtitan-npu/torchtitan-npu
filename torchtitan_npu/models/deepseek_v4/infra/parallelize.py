# Adapted from
# https://github.com/pytorch/torchtitan/blob/v0.2.1/torchtitan/models/deepseek_v3/infra/parallelize.py
# https://github.com/pytorch/torchtitan/blob/v0.2.1/torchtitan/models/llama4/infra/parallelize.py
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.config.job_config import Compile as CompileConfig
from torchtitan.distributed import NoParallel, ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.dual_pipe_v import (
    DualPipeExpertParallel,
    get_dual_pipe_v_flag,
)
from torchtitan.distributed.expert_parallel import (
    BaseExpertParallel,
    DeepEPExpertParallel,
    ExpertParallel,
    TensorParallel,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import apply_ddp
from torchtitan.models.llama4.infra.parallelize import apply_fsdp
from torchtitan.models.moe import moe as moe_module

from torchtitan_npu.converters.kernels.rms_norm import NPURMSNorm

from torchtitan_npu.models.deepseek_v4.model.model import (
    Attention,
    DSAIndexerLossLoggingHelper,
)


logger = logging.getLogger(__name__)


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops._c10d_functional.all_to_all_single.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch._higher_order_ops.inductor_compiled_code,
}


class AwaitRowwiseParallel(RowwiseParallel):
    @staticmethod
    def _prepare_output_fn(output_layouts, use_local_output, mod, outputs, device_mesh):
        # Rowwise sharding produces partial output, depending on output layouts:
        # 1. to replicate -> allreduce
        # 2. to shard -> reduce_scatter
        if outputs.placements != output_layouts:
            outputs = outputs.redistribute(placements=output_layouts, async_op=True)

        # wait for async redistribution to complete
        real_tensor = outputs._local_tensor
        torch.ops._c10d_functional.wait_tensor(real_tensor)

        # back to local tensor if use_local_output is True
        return outputs.to_local() if use_local_output else outputs


class PrepareModuleInputOutputWithBwdAllReduce(PrepareModuleInputOutput):
    """
    Extension of PrepareModuleInputOutput that registers backward hooks on specified inputs
    to perform allreduce on their gradients during backpropagation.

    This is useful when certain inputs participate in computations that require
    gradient synchronization across devices (e.g., in tensor parallelism scenarios).
    """

    def __init__(self, *, bwd_allreduce_inputs: tuple[bool, ...], **kwargs):
        super().__init__(**kwargs)
        self.bwd_allreduce_inputs = bwd_allreduce_inputs

        if self.prepare_module_input.input_layouts is not None:
            assert len(self.bwd_allreduce_inputs) == len(
                self.prepare_module_input.input_layouts
            ), (
                f"bwd_allreduce_inputs must have the same length as input_layouts! "
                f"Got {len(self.bwd_allreduce_inputs)} vs {len(self.prepare_module_input.input_layouts)}"
            )

    def _attach_bwd_hook_fn(self, module: nn.Module, inputs: tuple) -> None:
        """
        Register backward hooks on specified inputs to perform allreduce on gradients.

        Args:
            module: The module to register hooks on
            inputs: Tuple of input tensors to the module
        """
        for _, (inp, needs_allreduce) in enumerate(
            zip(inputs, self.bwd_allreduce_inputs)
        ):
            if not needs_allreduce:
                continue

            if not isinstance(inp, torch.Tensor) or not inp.requires_grad:
                continue

            def _allreduce_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                # Ensure gradient is contiguous for efficient communication
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                torch.distributed.all_reduce(
                    grad, op=torch.distributed.ReduceOp.SUM, group=self.group
                )
                return grad

            inp.register_hook(_allreduce_grad_hook)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        super()._apply(module, device_mesh)

        self.group = device_mesh.get_group()
        if self.prepare_module_input.use_local_output:
            module.register_forward_pre_hook(self._attach_bwd_hook_fn)

        return module


def _register_distributed_parameter(
    module: nn.Module,
    name: str,
    device_mesh: DeviceMesh,
    placements: list,
):
    dt = nn.Parameter(
        distribute_tensor(
            getattr(module, name),
            device_mesh=device_mesh,
            placements=placements,
            src_data_rank=0,
        )
    )
    module.register_parameter(name, dt)


def parallelize_deepseek_v4(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    attn_type = getattr(model.model_args, "attn_type", "sdpa")
    if job_config.parallelism.context_parallel_degree > 1 and attn_type != "sdpa":
        raise NotImplementedError(
            f"Context Parallel only supports SDPA attention. "
            f"Got attn_type='{attn_type}'. "
            f"FlexAttention and varlen attention are not supported with CP."
        )

    # patch the indexer loss tracking with distributed version to get the synchronized indexer loss metric
    apply_distributed_indexer_loss_tracking(parallel_dims)

    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        if enable_float8_tensorwise_tp:
            raise NotImplementedError(
                "Currently, float8 tensorwise TP is not tested for deepseekv3"
            )

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            job_config=job_config,
        )
        maybe_enable_async_tp(job_config, tp_mesh)

    # Check if using DeepEP for MoE communication
    if job_config.parallelism.expert_parallel_comm_backend == "deepep":
        if not parallel_dims.ep_enabled:
            raise ValueError(
                "DeepEP requires expert parallelism (ep_degree > 1). "
                "The DeepEP MoE model code does not support EP=1. "
                "Please set expert_parallel_degree > 1 or use standard communication backend."
            )
        if parallel_dims.etp_enabled:
            raise NotImplementedError(
                "DeepEP with Expert Tensor Parallelism (ETP) is not supported yet. "
                "Please set expert_tensor_parallel_degree=1 or use standard communication backend."
            )

        use_deepep = True

        # Import deepep module to register custom ops before accessing them
        import torchtitan.distributed.deepep  # noqa: F401 - registers torch.ops.deepep

        _op_sac_save_list.add(torch.ops.deepep.dispatch.default)
        _op_sac_save_list.add(torch.ops.deepep.combine.default)
    else:
        use_deepep = False

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        dual_pipe_v = get_dual_pipe_v_flag(job_config, parallel_dims)

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            dual_pipe_v=dual_pipe_v,
            use_deepep=use_deepep,
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, job_config.compile, parallel_dims.ep_enabled)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")

    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=model_compile_enabled,
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    job_config: JobConfig,
):
    """Apply tensor parallelism."""

    # whether the npu_dsa kernel is enabled
    parallel_cfg = job_config.parallelism
    use_cp = (
        parallel_cfg.enable_custom_context_parallel
        and parallel_cfg.context_parallel_degree > 1
    )
    enable_npu_dsa = "npu_dsa" in job_config.model.converters or use_cp
    enable_activation_checkpoint = job_config.activation_checkpoint.mode in [
        "full",
        "selective",
    ]

    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    (
        rowwise_parallel,
        await_rowwise_parallel,
        colwise_parallel,
        prepare_module_input,
        prepare_module_input_output,
    ) = (
        RowwiseParallel,
        AwaitRowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
        PrepareModuleInputOutput,
    )
    hc_head_plan = prepare_module_input_output(
        input_layouts=(Shard(1), Replicate(), Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        use_local_input=True,
        output_layouts=(Replicate()),
        desired_output_layouts=(Shard(1)),
        use_local_output=False,
    )
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
            "hc_head": hc_head_plan,
        },
    )
    _register_distributed_parameter(model, "hc_head_fn", tp_mesh, [Replicate()])
    _register_distributed_parameter(model, "hc_head_base", tp_mesh, [Replicate()])
    _register_distributed_parameter(model, "hc_head_scale", tp_mesh, [Replicate()])

    attention_kernel_plan_ratio1 = PrepareModuleInputOutputWithBwdAllReduce(
        bwd_allreduce_inputs=(False, True, False, False, False),
        input_layouts=(Shard(2), Replicate(), Shard(0), None, None),
        desired_input_layouts=(Shard(2), Replicate(), Shard(0), None, None),
        use_local_input=True,
        output_layouts=(Shard(2)),
        desired_output_layouts=(Shard(2)),
        use_local_output=False,
    )

    attention_kernel_plan_ratio4 = PrepareModuleInputOutputWithBwdAllReduce(
        bwd_allreduce_inputs=(False, True, False, True, False),
        input_layouts=(Shard(2), Replicate(), Shard(0), Replicate(), Replicate()),
        desired_input_layouts=(
            Shard(2),
            Replicate(),
            Shard(0),
            Replicate(),
            Replicate(),
        ),
        use_local_input=True,
        output_layouts=(Shard(2)),
        desired_output_layouts=(Shard(2)),
        use_local_output=False,
    )

    attention_kernel_plan_ratio128 = PrepareModuleInputOutputWithBwdAllReduce(
        bwd_allreduce_inputs=(False, True, False, True, False),
        input_layouts=(Shard(2), Replicate(), Shard(0), Replicate(), None),
        desired_input_layouts=(Shard(2), Replicate(), Shard(0), Replicate(), None),
        use_local_input=True,
        output_layouts=(Shard(2)),
        desired_output_layouts=(Shard(2)),
        use_local_output=False,
    )

    indexer_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate(), Replicate(), Replicate(), None),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate(), None),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate(), Replicate()),
        desired_output_layouts=(Replicate(), Replicate(), Replicate()),
        use_local_output=False,
    )

    li_compute_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate(), Replicate(), None, None),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), None, None),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate()),
        desired_output_layouts=(Replicate(), Replicate()),
        use_local_output=False,
    )

    compressor_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=False,
    )

    indexer_compressor_plan = prepare_module_input_output(
        input_layouts=(Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate()),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=True,
    )

    hc_pre_plan = prepare_module_input_output(
        input_layouts=(Shard(1), Replicate(), Replicate(), Replicate()),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        use_local_input=True,
        output_layouts=(Replicate(), Replicate(), Replicate()),
        desired_output_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=False,
    )

    hc_post_plan = prepare_module_input_output(
        input_layouts=(Shard(1), Shard(1), Shard(1), Shard(1)),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), Replicate()),
        use_local_input=True,
        output_layouts=(Replicate()),
        desired_output_layouts=(Shard(1)),
    )

    hc_pre_sinkhon_plan = prepare_module_input(
        input_layouts=(Replicate(), Replicate(), Replicate(), None, None, None),
        desired_input_layouts=(Replicate(), Replicate(), Replicate(), None, None, None),
        use_local_output=True,
    )

    get_window_topk_idxs_plan = prepare_module_input_output(
        input_layouts=(None, None, None),
        desired_input_layouts=(None, None, None),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=True,
    )

    get_compress_topk_idxs_plan = prepare_module_input_output(
        input_layouts=(Replicate(), None),
        desired_input_layouts=(Replicate(), None),
        use_local_input=False,
        output_layouts=(Replicate()),
        desired_output_layouts=(Replicate()),
        use_local_output=True,
    )

    li_loss_plan = prepare_module_input(
        input_layouts=(
            Shard(1),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            None,
            None,
        ),
        desired_input_layouts=(
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            Replicate(),
            None,
            None,
        ),
        use_local_output=True,
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for transformer_block in model.layers.values():
        _register_distributed_parameter(
            transformer_block.attention.inner_attention,
            "attn_sink",
            tp_mesh,
            [Shard(0)],
        )
        _register_distributed_parameter(
            transformer_block, "hc_attn_fn", tp_mesh, [Replicate()]
        )
        _register_distributed_parameter(
            transformer_block, "hc_ffn_fn", tp_mesh, [Replicate()]
        )
        _register_distributed_parameter(
            transformer_block, "hc_attn_base", tp_mesh, [Replicate()]
        )
        _register_distributed_parameter(
            transformer_block, "hc_ffn_base", tp_mesh, [Replicate()]
        )
        _register_distributed_parameter(
            transformer_block, "hc_attn_scale", tp_mesh, [Replicate()]
        )
        _register_distributed_parameter(
            transformer_block, "hc_ffn_scale", tp_mesh, [Replicate()]
        )

        if transformer_block.attention.compress_ratio == 1:
            attention_kernel_plan = attention_kernel_plan_ratio1
        elif transformer_block.attention.compress_ratio == 4:
            attention_kernel_plan = attention_kernel_plan_ratio4
        else:
            attention_kernel_plan = attention_kernel_plan_ratio128

        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), Replicate(), None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None),
            ),
            "attention.inner_attention.sparse_attn.get_window_topk_idxs": get_window_topk_idxs_plan,
            "attention.inner_attention.sparse_attn.get_compress_topk_idxs": get_compress_topk_idxs_plan,
            # NOTE: use_local_output=False make the output to be a DTensor instead of a plain Tensor
            # so that the intermedidate results k is generated as a DTensor and its gradient is
            # correctly handled by the autograd engine.
            "attention.pre_attention.wq_a": NoParallel(use_local_output=False),
            "attention.pre_attention.q_norm": NoParallel(use_local_output=False),
            "attention.pre_attention.wq_b": colwise_parallel(use_local_output=False),
            "attention.pre_attention.wkv": NoParallel(use_local_output=False),
            "attention.pre_attention.kv_norm": NoParallel(use_local_output=False),
            "attention.post_attention.wo_a": colwise_parallel(use_local_output=False),
            "attention.post_attention.wo_b": rowwise_parallel(
                input_layouts=Shard(-1),
                output_layouts=Shard(1),
                use_local_output=False,
            ),
            "attention.inner_attention.sparse_attn": attention_kernel_plan,
            "hc_post": hc_post_plan,
            "hc_pre": hc_pre_plan,
            "hc_pre.torch_hc_split_sinkhorn": hc_pre_sinkhon_plan,
            "cal_index_loss.li_loss": li_loss_plan,
            "ffn_norm": SequenceParallel(),
        }
        if transformer_block.attention.compress_ratio > 1:
            compress_ratio = transformer_block.attention.compress_ratio
            if compress_ratio == 4:
                compressor_attr = "compressor"
            else:
                compressor_attr = "compressor_128"
            compressor_module = getattr(
                transformer_block.attention.pre_attention, compressor_attr
            )
            compressor_key = f"attention.pre_attention.{compressor_attr}"
            _register_distributed_parameter(
                compressor_module, "ape", tp_mesh, [Replicate()]
            )
            layer_plan.update(
                {
                    compressor_key: compressor_plan,
                    f"{compressor_key}.wkv": NoParallel(use_local_output=False),
                    f"{compressor_key}.wgate": NoParallel(use_local_output=False),
                    f"{compressor_key}.norm": NoParallel(use_local_output=False),
                }
            )
            if compress_ratio == 4:
                _register_distributed_parameter(
                    transformer_block.attention.pre_attention.indexer.compressor,
                    "ape",
                    tp_mesh,
                    [Replicate()],
                )
                layer_plan.update(
                    {
                        "attention.inner_attention.li_compute": li_compute_plan,
                        "attention.pre_attention.indexer": indexer_plan,
                        "attention.pre_attention.indexer.compressor": indexer_compressor_plan,
                        "attention.pre_attention.indexer.wq_b": NoParallel(
                            use_local_output=True
                        ),
                        "attention.pre_attention.indexer.weights_proj": NoParallel(
                            use_local_output=True
                        ),
                        "attention.pre_attention.indexer.compressor.wkv": NoParallel(
                            use_local_output=False
                        ),
                        "attention.pre_attention.indexer.compressor.wgate": NoParallel(
                            use_local_output=False
                        ),
                        "attention.pre_attention.indexer.compressor.norm": NoParallel(
                            use_local_output=False
                        ),
                    }
                )

        if not transformer_block.moe_enabled:
            # Select the appropriate parallel strategy:
            # Use AwaitRowwiseParallel when activation checkpoint is enabled to handle
            # async redistribution. The custom implementation ensures wait_tensor() is called
            # on _local_tensor to prevent memory leaks caused by incomplete async operations.
            safe_rowwise_parallel = (
                await_rowwise_parallel
                if enable_activation_checkpoint
                else rowwise_parallel
            )
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": safe_rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        if transformer_block.layer_id >= model.model_args.n_layers:
            layer_plan.update(
                {
                    "enorm": SequenceParallel(),
                    "hnorm": SequenceParallel(),
                    "e_proj": SequenceParallel(use_local_output=True),
                    "h_proj": SequenceParallel(use_local_output=True),
                }
            )

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    dual_pipe_v: bool = False,
    use_deepep: bool = False,
):
    assert (
        tp_mesh is not None or ep_mesh is not None
    ), f"""
        At least one of Tensor Parallel mesh (tp_mesh) or Expert Parallel mesh (ep_mesh) must be provided.
        Current status: tp_mesh={tp_mesh}, ep_mesh={ep_mesh}
        """

    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1), Replicate()),
                    desired_input_layouts=(Shard(1), Shard(1)),
                    use_local_input=True,
                    output_layouts=(Shard(1),),
                    desired_output_layouts=(Shard(1),),
                ),
                "moe.router.gate": SequenceParallel(
                    sequence_dim=0, use_local_output=True
                ),
            }
            if transformer_block.moe.shared_experts is not None:
                # input: sharded on fused batch-seq dimension (dim=0)
                # all-gather for input, reduce-scatter for output
                moe_layer_plan.update(
                    {
                        "moe.shared_experts": PrepareModuleInput(
                            input_layouts=(Shard(0),),
                            desired_input_layouts=(Replicate(),),
                        ),
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Shard(0)
                        ),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        # Currently only TP and TP extend EP are supported
        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp_mesh is None:
            experts_mesh = ep_mesh
            if use_deepep:
                score_before_experts = transformer_block.moe.score_before_experts
                experts_plan = DeepEPExpertParallel(
                    score_before_experts=score_before_experts,
                )
                logger.info("Applying DeepEP to MoE layer")
            else:
                experts_plan = ExpertParallel()
        else:
            raise NotImplementedError("ETP is not supported currently")

        if dual_pipe_v and isinstance(experts_plan, BaseExpertParallel):
            experts_plan = DualPipeExpertParallel(experts_plan)

        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def apply_compile(model: nn.Module, compile_config: CompileConfig, ep_enabled: bool):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    # but it is experimental.
    torch._dynamo.config.capture_scalar_outputs = True
    # Workaround for https://github.com/pytorch/pytorch/issues/166926
    # pyrefly: ignore [missing-attribute]
    torch._C._dynamo.eval_frame._set_lru_cache(False)
    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.named_children():
        if transformer_block.moe_enabled:
            # If it is a MoE layer, FSDP(GroupedExperts) will cause a graph break
            # So we must weave compile wrappers around those FSDP hooks to
            # prevent AC from falling back the whole graph to eager.
            # TODO: Fix Compile(AC(graph break))

            if isinstance(transformer_block, CheckpointWrapper):
                # TODO: Make CheckpointWrapper a transparent wrapper
                # unwrap so that .named_children() works
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            for attr_name, submod in block.named_children():
                assert getattr(block, attr_name) == getattr(
                    transformer_block, attr_name
                )

                if attr_name in {"cal_index_loss", "hc_pre"}:
                    continue

                if isinstance(submod, moe_module.MoE):
                    # avoid graph breaking on the GroupedExperts' FSDP hooks
                    # by wrapping each submod's forward instead of their __call__
                    moe = submod
                    for attr_name, submod in moe.named_children():
                        if attr_name == "experts":
                            # NOTE: We don't compile token dispatch and token combine due to an issue on B200:
                            # https://github.com/pytorch/torchtitan/issues/1940
                            continue
                        setattr(
                            moe,
                            attr_name,
                            torch.compile(
                                submod, backend=compile_config.backend, fullgraph=True
                            ),
                        )
                elif isinstance(submod, Attention):
                    # inner_attention contains NPU fused ops (sparse_attn, li_compute) that cannot be compiled.
                    # Compile pre_attention and post_attention as whole units for better efficiency.
                    attention = submod
                    for attr_name, submod in attention.named_children():
                        if attr_name == "inner_attention":
                            continue
                        setattr(
                            attention,
                            attr_name,
                            torch.compile(
                                submod, backend=compile_config.backend, fullgraph=True
                            ),
                        )
                else:
                    setattr(
                        block,
                        attr_name,
                        torch.compile(
                            submod, backend=compile_config.backend, fullgraph=True
                        ),
                    )

        else:
            # If it's not a MoE layer, there is no FSDP(GroupedExperts)
            # So we can compile the whole block
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )

        # pyrefly: ignore [missing-attribute]
        model.layers.register_module(layer_id, transformer_block)

    # Patch some globals only once (apply_compile is called multiple times for PP setup)
    already_patched = (
        "_run_experts_grouped_mm_dynamic"
        in moe_module._run_experts_grouped_mm.__qualname__
    )
    if not already_patched:
        moe_module._run_experts_grouped_mm = torch.compile(
            moe_module._run_experts_grouped_mm,
            backend=compile_config.backend,
            fullgraph=True,
        )

        if ep_enabled:
            compiled_fn = moe_module._run_experts_grouped_mm

            # keep function logic in sync with `already_patched` above
            def _run_experts_grouped_mm_dynamic(
                w1: torch.Tensor,
                w2: torch.Tensor,
                w3: torch.Tensor,
                x: torch.Tensor,
                num_tokens_per_expert: torch.Tensor,
            ) -> torch.Tensor:
                # dynamic number of tokens in expert parallel
                torch._dynamo.mark_dynamic(x, 0)
                return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

            moe_module._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic

    # NOTE: We don't compile for loop code path due to an issue with unbacked symints:
    # https://github.com/pytorch/pytorch/issues/166460

    logger.info("Compiling each TransformerBlock with torch.compile")


def apply_distributed_indexer_loss_tracking(parallel_dims: ParallelDims):
    """
    Dynamically patch track_dsa_indexer_metrics to support 3D/4D parallelism
    synchronization efficiently using a single global communication step.

    Before synchronization, the indexer loss on each GPU is merely an average
    over its local [B, S] (Batch, Sequence) shape. In a distributed scenario,
    this local loss must be synchronized (averaged) across all parallel domains,
    including Pipeline Parallel (PP), Tensor Parallel (TP), Data Parallel (DP),
    and Context Parallel (CP) groups, to obtain the globally accurate metric.
    """

    @staticmethod
    def distributed_track_dsa_indexer_metrics(total_acc_steps: int):
        tracker = DSAIndexerLossLoggingHelper.tracker
        if "values" not in tracker:
            return

        # 1. Clone the tensor to avoid modifying the underlying tracker.
        # Shape is always [total_num_layers], so tensor size is consistent across all ranks.
        dsa_indexer_losses = tracker["values"].clone()

        if dist.is_initialized():
            # 2. Perform a SINGLE global All-Reduce (AVG).
            # This averages the tensor across all ranks in the world.
            # For any specific layer, only the ranks in its corresponding PP stage have non-zero values.
            # Therefore, the global sum for a layer is exactly the sum across its non-PP domains.
            dist.all_reduce(dsa_indexer_losses, op=dist.ReduceOp.AVG)

            # 3. Correct the mathematical expectation for Pipeline Parallelism (PP).
            # A global AVG divides the sum by WORLD_SIZE.
            # However, the valid non-zero values only come from (WORLD_SIZE // PP_DEGREE) ranks.
            # By multiplying the result by PP_DEGREE, we recover the exact mathematical
            # average across the non-PP domains (FSDP, TP, CP, etc.).
            pp_degree = parallel_dims.pp if parallel_dims.pp_enabled else 1
            dsa_indexer_losses *= pp_degree

        # 4. Calculate the final aggregated loss.
        # Divide by total gradient accumulation steps to get the true average per step.
        dsa_indexer_num_layers = torch.count_nonzero(dsa_indexer_losses).item()
        loss = dsa_indexer_losses.sum() / dsa_indexer_num_layers / total_acc_steps

        # 5. Clean the tracker and log the metric
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        logger.info(f"indexer loss: {loss.item()}")

    # Apply the monkey patch
    DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics = (
        distributed_track_dsa_indexer_metrics
    )
