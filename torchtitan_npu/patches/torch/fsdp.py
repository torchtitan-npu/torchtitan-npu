from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp


def _get_gradient_divide_factors(
    reduce_scatter_group: Optional[dist.ProcessGroup],
    all_reduce_group: Optional[dist.ProcessGroup],
    reduce_dtype: torch.dtype,
    device_type: str = "",
    factor: Optional[float] = None,
    force_sum_reduction_for_comms: bool = False,
) -> tuple[
    Optional[float],
    Optional[float],
    Union[dist.ReduceOp, dist.ReduceOp.RedOpType],
    Union[dist.ReduceOp, dist.ReduceOp.RedOpType],
]:
    # MTIA appears to only support SUM reduction, hence we force it implicitly
    if device_type == "mtia":
        force_sum_reduction_for_comms = True

    # For fp32/bf16, we do not need to worry about overflow/underflow, so we
    # use NCCL's built-in division to avoid separate div kernels
    overflow_risk = reduce_dtype not in (torch.float32, torch.bfloat16)
    if reduce_scatter_group is not None:
        data_parallel_size = reduce_scatter_group.size()
    else:
        data_parallel_size = 1

    if all_reduce_group is not None:
        data_parallel_size *= all_reduce_group.size()

    if not overflow_risk and not force_sum_reduction_for_comms:
        if factor is None:
            # Warning: NCCL ReduceOp.AVG may produce incorrect results with
            # world size 1.
            if data_parallel_size == 1:
                return None, None, ReduceOp.SUM, ReduceOp.SUM
            return None, None, ReduceOp.AVG, ReduceOp.AVG
        if reduce_scatter_group is not None and factor == reduce_scatter_group.size():
            reduce_scatter_op = ReduceOp.AVG
        else:
            reduce_scatter_op = torch.distributed._make_nccl_premul_sum(1 / factor)
        return None, None, ReduceOp.SUM, ReduceOp.SUM       # NOTE: npu patch, npu only supports SUM

    if factor is None:
        factor = float(data_parallel_size)
    pre_factor: Optional[float]
    if overflow_risk:
        # Since fp16 has smaller dynamic range than fp32/bf16, we want to avoid
        # overflow/underflow. For N data parallel workers, each worker computes
        # g_i, and they collectively reduce (g_1 + ... + g_N) / N. To avoid
        # overflow/underflow, we divide by ~sqrt(N) before/after the reduction.
        pre_factor = 1
        while factor % pre_factor == 0 and factor / pre_factor > pre_factor:
            pre_factor *= 2
        post_factor = factor / pre_factor
    else:
        # Prefer post-multiplying as it operates on less data and is thus faster
        pre_factor, post_factor = None, factor

    return pre_factor, post_factor, ReduceOp.SUM, ReduceOp.SUM


# apply patch
torch.distributed.fsdp._fully_shard._fsdp_collectives._get_gradient_divide_factors = _get_gradient_divide_factors