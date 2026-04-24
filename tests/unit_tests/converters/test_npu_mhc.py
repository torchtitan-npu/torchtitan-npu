# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchtitan_npu.converters.kernels.triton_op_code.add import add_fwd
from torchtitan_npu.converters.kernels.triton_op_code.mhc_triton import (
    MHCPostTriton,
    MHCPreOnlyTriton,
    MHCPreTriton,
)


# =============================================================================
# Torch reference implementation for accuracy comparison
# =============================================================================
def sinkhorn_knopps(h_res, sinkhorn_iters, eps):
    h_res = h_res.softmax(-1) + eps
    col_sum = h_res.sum(-2, keepdim=True)
    h_res = h_res / (col_sum + eps)
    for _ in range(sinkhorn_iters - 1):
        row_sum = h_res.sum(-1, keepdim=True)
        h_res = h_res / (row_sum + eps)
        col_sum = h_res.sum(-2, keepdim=True)
        h_res = h_res / (col_sum + eps)
    return h_res


def hc_split_sinkhorn_torch(
    weight: torch.Tensor,
    branch_alpha: torch.Tensor,
    branch_beta: torch.Tensor,
    num_stream: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
):
    h_pre, h_post, h_res = weight.split(
        [num_stream, num_stream, num_stream * num_stream], dim=-1
    )
    h_res = h_res.unflatten(-1, (num_stream, num_stream))

    h_pre = (
        F.sigmoid(
            h_pre * branch_alpha[0] + branch_beta[:num_stream].unsqueeze(0).unsqueeze(0)
        )
        + eps
    )
    h_post = 2 * F.sigmoid(
        h_post * branch_alpha[1]
        + branch_beta[num_stream : 2 * num_stream].unsqueeze(0).unsqueeze(0)
    )
    h_res = h_res * branch_alpha[2] + branch_beta[2 * num_stream :].view(
        num_stream, num_stream
    ).unsqueeze(0).unsqueeze(0)

    h_res = sinkhorn_knopps(h_res, sinkhorn_iters, eps)
    return h_pre, h_post, h_res


class MhcModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_stream = 4
        self.hc_eps = 1e-6
        self.norm_eps = 1e-6

    def hc_pre(
        self,
        x: torch.Tensor,
        phi_weight,
        branch_alpha,
        branch_beta,
        norm_gamma,
        mhc_use_gamma=True,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.float()
        phi_weight = phi_weight.float().t()
        branch_alpha = branch_alpha.float()
        branch_beta = branch_beta.float()
        norm_gamma = norm_gamma.float()

        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        if mhc_use_gamma:
            x_normed = x * rsqrt * norm_gamma
        else:
            x_normed = x * rsqrt

        weight = torch.matmul(x_normed, phi_weight)
        h_pre, h_post, h_res = hc_split_sinkhorn_torch(
            weight, branch_alpha, branch_beta, self.num_stream
        )
        y = torch.sum(
            h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(self.num_stream, -1)),
            dim=2,
        )
        return y.to(dtype), h_post, h_res

    def hc_post(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
        h_post: torch.Tensor,
        h_res: torch.Tensor,
    ):
        y = (
            h_post.unsqueeze(-1) * x.unsqueeze(-2)
            + torch.sum(
                h_res.unsqueeze(-1)
                * residual.unflatten(dim=-1, sizes=(self.num_stream, -1)).unsqueeze(-2),
                dim=2,
            )
        ).flatten(2)
        return y.type_as(x)

    def hc_pre_only(
        self,
        x,
        phi_weight,
        branch_alpha,
        branch_beta,
        norm_gamma,
        mhc_use_gamma=True,
        num_stream=4,
        sinkhorn_iters=20,
        eps=1e-6,
    ):
        shape, dtype = x.size(), x.dtype
        x = x.float()
        phi_weight = phi_weight.float().t()
        branch_alpha = branch_alpha.float()
        branch_beta = branch_beta.float()
        norm_gamma = norm_gamma.float()

        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)
        if mhc_use_gamma:
            weight = torch.matmul(x * rsqrt * norm_gamma, phi_weight)
        else:
            weight = torch.matmul(x, phi_weight) * rsqrt
        h_pre = (
            F.sigmoid(weight * branch_alpha + branch_beta.unsqueeze(0).unsqueeze(0))
            + eps
        )
        y = torch.sum(
            h_pre.unsqueeze(-1) * x.unflatten(dim=-1, sizes=(num_stream, -1)), dim=2
        )
        return y.to(dtype)


# =============================================================================
# test_add
# =============================================================================
@pytest.mark.parametrize(
    ("B", "S", "N", "D"),
    [
        pytest.param(*test, id="B{}-S{}-N{}-D{}".format(*test))
        for test in [(1, 1024, 4, 8192), (1, 2048, 4, 4096), (1, 4096, 4, 4096)]
    ],
)
def test_add_triton(B, S, N, D):
    device = "npu:0"
    data_type = torch.float32

    x = torch.rand((B * S, N * D), dtype=data_type, requires_grad=True).npu()
    y = torch.rand((B * S, N * D), dtype=data_type, requires_grad=True).npu()

    out_triton = add_fwd(x, y)
    out_torch = x + y

    torch.testing.assert_close(
        out_triton, out_torch, rtol=1e-3, atol=1e-3, equal_nan=True
    )


# =============================================================================
# test_mhc_only_pre
# =============================================================================
@pytest.mark.parametrize(
    ("B", "S", "N", "D", "mhc_use_gamma"),
    [
        pytest.param(*test, id="B{}-S{}-N{}-D{}-mhc_use_gamma{}".format(*test))
        for test in [
            (1, 1024, 4, 8192, True),
            (1, 2048, 4, 4096, True),
            (1, 4096, 4, 4096, True),
            (1, 1024, 4, 8192, False),
            (1, 2048, 4, 4096, False),
            (1, 4096, 4, 4096, False),
        ]
    ],
)
def test_mhc_pre_only_triton(B, S, N, D, mhc_use_gamma):
    torch.manual_seed(42)
    device = "npu:0"
    data_type = torch.float32

    x = torch.rand(B, S, N * D, device="cpu", dtype=data_type, requires_grad=True).npu()
    x_torch = x.clone().detach().requires_grad_(True)
    x_triton = x.clone().detach().requires_grad_(True)
    weight = torch.rand(
        N, N * D, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    weight_torch = weight.clone().detach().requires_grad_(True)
    weight_triton = weight.clone().detach().requires_grad_(True)
    branch_alpha = torch.rand(
        1, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    branch_alpha_torch = branch_alpha.clone().detach().requires_grad_(True)
    branch_alpha_triton = branch_alpha.clone().detach().requires_grad_(True)
    branch_beta = torch.rand(N, device="cpu", dtype=data_type, requires_grad=True).npu()
    branch_beta_torch = branch_beta.clone().detach().requires_grad_(True)
    branch_beta_triton = branch_beta.clone().detach().requires_grad_(True)
    norm_gamma = torch.rand(
        N * D, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    norm_gamma_torch = norm_gamma.clone().detach().requires_grad_(True)
    norm_gamma_triton = norm_gamma.clone().detach().requires_grad_(True)

    mhc_torch = MhcModule()
    y_torch = mhc_torch.hc_pre_only(
        x_torch,
        weight_torch,
        branch_alpha_torch,
        branch_beta_torch,
        norm_gamma_torch,
        mhc_use_gamma,
    )

    y_triton = MHCPreOnlyTriton.apply(
        x_triton,
        weight_triton,
        branch_alpha_triton,
        branch_beta_triton,
        norm_gamma_triton,
        mhc_use_gamma,
        1e-6,
        N,
    )

    rtol = 1e-2
    atol = 1e-2

    torch.testing.assert_close(y_torch, y_triton, rtol=rtol, atol=atol, equal_nan=True)

    loss_torch = y_torch.sum()
    loss_torch.backward()

    loss_triton = y_triton.sum()
    loss_triton.backward()

    torch.testing.assert_close(
        x_torch.grad, x_triton.grad, rtol=rtol, atol=atol, equal_nan=True
    )
    torch.testing.assert_close(
        weight_torch.grad, weight_triton.grad, rtol=rtol, atol=atol, equal_nan=True
    )
    torch.testing.assert_close(
        branch_alpha_torch.grad,
        branch_alpha_triton.grad,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    torch.testing.assert_close(
        branch_beta_torch.grad,
        branch_beta_triton.grad,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    if mhc_use_gamma:
        torch.testing.assert_close(
            norm_gamma_torch.grad,
            norm_gamma_triton.grad,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )


# =============================================================================
# test_mhc_post
# =============================================================================
@pytest.mark.parametrize(
    ("B", "S", "N", "D"),
    [
        pytest.param(*test, id="B{}-S{}-N{}-D{}".format(*test))
        for test in [(1, 2048, 4, 4096), (1, 4096, 4, 4096)]
    ],
)
def test_mhc_post_triton(B, S, N, D):
    torch.manual_seed(42)
    device = "npu:0"
    data_type = torch.float32

    x = torch.rand(B, S, D, device="cpu", dtype=data_type, requires_grad=True).npu()
    x_torch = x.clone().detach().requires_grad_(True)
    x_triton = x.clone().detach().requires_grad_(True)
    residual = torch.rand(
        B, S, N * D, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    residual_torch = residual.clone().detach().requires_grad_(True)
    residual_triton = residual.clone().detach().requires_grad_(True)
    h_post = torch.rand(
        B, S, N, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    h_post_torch = h_post.clone().detach().requires_grad_(True)
    h_post_triton = h_post.clone().detach().requires_grad_(True)
    h_res = torch.rand(
        B, S, N, N, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    h_res_torch = h_res.clone().detach().requires_grad_(True)
    h_res_triton = h_res.clone().detach().requires_grad_(True)

    mhc_torch = MhcModule()
    result_torch = mhc_torch.hc_post(x_torch, residual_torch, h_post_torch, h_res_torch)

    result_triton = MHCPostTriton.apply(
        x_triton, residual_triton, h_post_triton, h_res_triton
    )

    rtol = 1e-2
    atol = 1e-2

    torch.testing.assert_close(
        result_torch, result_triton, rtol=rtol, atol=atol, equal_nan=True
    )

    loss_torch = result_torch.sum()
    loss_torch.backward()

    loss_triton = result_triton.sum()
    loss_triton.backward()

    torch.testing.assert_close(
        x_torch.grad, x_triton.grad, rtol=rtol, atol=atol, equal_nan=True
    )
    torch.testing.assert_close(
        residual_torch.grad, residual_triton.grad, rtol=rtol, atol=atol, equal_nan=True
    )
    torch.testing.assert_close(
        h_post_torch.grad, h_post_triton.grad, rtol=rtol, atol=atol, equal_nan=True
    )
    torch.testing.assert_close(
        h_res_torch.grad, h_res_triton.grad, rtol=rtol, atol=atol, equal_nan=True
    )


# =============================================================================
# test_mhc_pre
# =============================================================================
@pytest.mark.parametrize(
    ("B", "S", "N", "D", "mhc_use_gamma"),
    [
        pytest.param(*test, id="B{}-S{}-N{}-D{}-mhc_use_gamma{}".format(*test))
        for test in [
            (1, 1024, 4, 8192, True),
            (1, 2048, 4, 4096, True),
            (1, 4096, 4, 4096, True),
            (1, 1024, 4, 8192, False),
            (1, 2048, 4, 4096, False),
            (1, 4096, 4, 4096, False),
        ]
    ],
)
def test_mhc_pre_triton(B, S, N, D, mhc_use_gamma):
    torch.manual_seed(42)
    device = "npu:0"
    data_type = torch.float32

    x = torch.rand(B, S, N * D, device="cpu", dtype=data_type, requires_grad=True).npu()
    x_torch = x.clone().detach().requires_grad_(True)
    x_triton = x.clone().detach().requires_grad_(True)
    weight = torch.rand(
        N * N + 2 * N, N * D, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    weight_torch = weight.clone().detach().requires_grad_(True)
    weight_triton = weight.clone().detach().requires_grad_(True)
    branch_alpha = torch.rand(
        3, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    branch_alpha_torch = branch_alpha.clone().detach().requires_grad_(True)
    branch_alpha_triton = branch_alpha.clone().detach().requires_grad_(True)
    branch_beta = torch.rand(
        2 * N + N * N, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    branch_beta_torch = branch_beta.clone().detach().requires_grad_(True)
    branch_beta_triton = branch_beta.clone().detach().requires_grad_(True)
    norm_gamma = torch.rand(
        N * D, device="cpu", dtype=data_type, requires_grad=True
    ).npu()
    norm_gamma_torch = norm_gamma.clone().detach().requires_grad_(True)
    norm_gamma_triton = norm_gamma.clone().detach().requires_grad_(True)

    mhc_torch = MhcModule()
    y_torch, h_post_torch, h_res_torch = mhc_torch.hc_pre(
        x_torch,
        weight_torch,
        branch_alpha_torch,
        branch_beta_torch,
        norm_gamma_torch,
        mhc_use_gamma,
    )

    y_triton, h_post_triton, h_res_triton = MHCPreTriton.apply(
        x_triton,
        weight_triton,
        branch_alpha_triton,
        branch_beta_triton,
        norm_gamma_triton,
        mhc_use_gamma,
        N,
        20,
        1e-6,
    )

    rtol = 1e-2
    atol = 1e-2

    torch.testing.assert_close(y_torch, y_triton, rtol=rtol, atol=atol, equal_nan=True)
    torch.testing.assert_close(
        h_post_torch, h_post_triton, rtol=rtol, atol=atol, equal_nan=True
    )
    torch.testing.assert_close(
        h_res_torch, h_res_triton, rtol=rtol, atol=atol, equal_nan=True
    )

    loss_torch = y_torch.sum()
    loss_torch.backward()

    loss_triton = y_triton.sum()
    loss_triton.backward()

    torch.testing.assert_close(
        x_torch.grad, x_triton.grad, rtol=rtol, atol=atol, equal_nan=True
    )
    torch.testing.assert_close(
        weight_torch.grad, weight_triton.grad, rtol=rtol, atol=atol, equal_nan=True
    )
    torch.testing.assert_close(
        branch_alpha_torch.grad,
        branch_alpha_triton.grad,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    torch.testing.assert_close(
        branch_beta_torch.grad,
        branch_beta_triton.grad,
        rtol=rtol,
        atol=atol,
        equal_nan=True,
    )
    if mhc_use_gamma:
        torch.testing.assert_close(
            norm_gamma_torch.grad,
            norm_gamma_triton.grad,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
