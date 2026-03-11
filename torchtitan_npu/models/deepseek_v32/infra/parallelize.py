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
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import NoParallel, ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.dual_pipe_v import DualPipeExpertParallel, get_dual_pipe_v_flag
from torchtitan.distributed.expert_parallel import (
    BaseExpertParallel,
    DeepEPExpertParallel,
    ExpertParallel,
    TensorParallel,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import apply_ddp
from torchtitan.models.llama4.infra.parallelize import (
    apply_compile,
    apply_fsdp,
)

from torchtitan_npu.converters.kernels.dsa import SparseLightningIndexerKLLoss
from torchtitan_npu.models.deepseek_v32.model.model import DSAIndexerLossLoggingHelper


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


class PrepareModuleInputOutputWithBwdAllReduce(PrepareModuleInputOutput):
    """
    Extension of PrepareModuleInputOutput that registers backward hooks on specified inputs
    to perform allreduce on their gradients during backpropagation.

    This is useful when certain inputs participate in computations that require
    gradient synchronization across devices (e.g., in tensor parallelism scenarios).
    """
    def __init__(
        self,
        *,
        bwd_allreduce_inputs: tuple[bool, ...],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bwd_allreduce_inputs = bwd_allreduce_inputs

        if self.prepare_module_input.input_layouts is not None:
            assert len(self.bwd_allreduce_inputs) == len(self.prepare_module_input.input_layouts), (
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
        for _, (inp, needs_allreduce) in enumerate(zip(inputs, self.bwd_allreduce_inputs)):
            if not needs_allreduce:
                continue

            if not isinstance(inp, torch.Tensor) or not inp.requires_grad:
                continue

            def _allreduce_grad_hook(grad: torch.Tensor) -> torch.Tensor:
                # Ensure gradient is contiguous for efficient communication
                if not grad.is_contiguous():
                    grad = grad.contiguous()
                torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, group=self.group)
                return grad

            inp.register_hook(_allreduce_grad_hook)
    
    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        super()._apply(module, device_mesh)

        self.group = device_mesh.get_group()
        if self.prepare_module_input.use_local_output:
            module.register_forward_pre_hook(
                self._attach_bwd_hook_fn
            )

        return module


def parallelize_deepseekv32(
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
            # pyrefly: ignore [bad-argument-type]
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
    job_config: JobConfig
):
    """Apply tensor parallelism."""

    # whether the npu_dsa kernel is enabled
    parallel_cfg = job_config.parallelism
    use_cp = parallel_cfg.enable_custom_context_parallel and parallel_cfg.context_parallel_degree > 1
    enable_npu_dsa = "npu_dsa" in job_config.model.converters or use_cp
    enable_mla_absorb = getattr(model.model_args, "enable_mla_absorb", True)

    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
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
        },
    )

    rowwise_parallel, colwise_parallel, prepare_module_input, prepare_module_input_output = (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
        PrepareModuleInputOutput,
    )

    if enable_mla_absorb:
        attention_kernel_plan = PrepareModuleInputOutputWithBwdAllReduce(
            bwd_allreduce_inputs=(False, True, True),
            input_layouts=(Shard(1), Replicate(), Replicate()),
            desired_input_layouts=(Shard(1), Replicate(), Replicate()),
            use_local_input=True,
            output_layouts=(Replicate(), Shard(1)),
            desired_output_layouts=(Replicate(), Shard(1)),
            use_local_output=False,
        )
    else:
        attention_kernel_plan = prepare_module_input_output(
            input_layouts=(Shard(1), Replicate(), Replicate()),
            desired_input_layouts=(Shard(1), Replicate(), Replicate()),
            use_local_input=True,
            output_layouts=(Replicate(), Shard(1)),
            desired_output_layouts=(Replicate(), Shard(1)),
            use_local_output=False,
        )

    indexer_plan = prepare_module_input(
        input_layouts=(Replicate(), Replicate(), None, Replicate(), None),
        desired_input_layouts=(Replicate(), Replicate(), None, Replicate(), None),
        use_local_output=True,
    )

    if enable_npu_dsa:
        # for SparseLightningIndexerKLLoss.forward
        # do allgather for query and softmax_max/sum, then indexer_loss on each tp_rank of a tp_group is the same
        indexer_loss_plan = prepare_module_input(
            input_layouts=(Shard(2),) + (Replicate(),) * 5 + (Shard(3),) * 2,
            desired_input_layouts=(Replicate(),) * 8,
            input_kwarg_layouts={"query_rope": Shard(2)},
            desired_input_kwarg_layouts={"query_rope": Replicate()},
            use_local_output=True,
        )
    else:
        # for DSAIndexerLoss.forward
        # do allreduce for selected_main_attn_dist, then indexer_loss on each tp_rank of a tp_group is the same
        indexer_loss_plan = prepare_module_input(
            input_layouts=(Partial(), Replicate(), Replicate(), None),
            desired_input_layouts=(Replicate(), Replicate(), Replicate(), None),
            use_local_output=True,
        )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    for transformer_block in model.layers.values():
        if enable_npu_dsa:
            # NOTE: here we patch the indexer_loss computation with npu fusion kernel module
            #       then we set the specific parallelize_plan for this module to ensure the correctness of loss
            transformer_block.attention.inner_attention.compute_dsa_indexer_loss = SparseLightningIndexerKLLoss()

        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), Replicate(), None, None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None, None),
            ),
            # NOTE: use_local_output=False make the output to be a DTensor instead of a plain Tensor
            # so that the intermedidate results k is generated as a DTensor and its gradient is
            # correctly handled by the autograd engine.
            "attention.wkv_a": NoParallel(use_local_output=False),
            "attention.wkv_b": colwise_parallel(use_local_output=False),
            "attention.kv_norm": NoParallel(use_local_output=False),
            # the indxer module params are not parallelized
            "attention.indexer": indexer_plan,
            "attention.indexer.wq_b": NoParallel(use_local_output=True),
            "attention.indexer.wk": NoParallel(use_local_output=True),
            "attention.indexer.k_norm": NoParallel(use_local_output=True),
            "attention.indexer.weights_proj": NoParallel(use_local_output=True),
            "attention.inner_attention": attention_kernel_plan,
            "attention.inner_attention.compute_dsa_indexer_loss": indexer_loss_plan,
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }

        if transformer_block.attention.q_lora_rank == 0:
            layer_plan.update(
                {
                    "attention.wq": colwise_parallel(
                        use_local_output=False
                    ),  # This is only used when q_lora_rank==0
                }
            )
        else:
            layer_plan.update(
                {
                    "attention.wq_a": NoParallel(use_local_output=False),
                    "attention.wq_b": colwise_parallel(use_local_output=False),
                    "attention.q_norm": NoParallel(use_local_output=False),
                }
            )

        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
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
    use_deepep: bool = False
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
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Shard(1),),
                    use_local_input=True,
                    output_layouts=(Shard(1),),
                    desired_output_layouts=(Shard(1),),
                ),
                "moe.router.gate": SequenceParallel(sequence_dim=0, use_local_output=True),
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
                        "moe.shared_experts.w2": RowwiseParallel(output_layouts=Shard(0)),
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
        dsa_indexer_num_layers = dsa_indexer_losses.shape[0]
        loss = dsa_indexer_losses.sum() / dsa_indexer_num_layers / total_acc_steps

        # 5. Clean the tracker and log the metric
        DSAIndexerLossLoggingHelper.clean_loss_in_tracker()
        logger.info(f"indexer loss: {loss.item()}")

    # Apply the monkey patch
    DSAIndexerLossLoggingHelper.track_dsa_indexer_metrics = distributed_track_dsa_indexer_metrics