# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


def _gather_freqs_cis_by_positions_batched(
    freqs_cis: torch.Tensor, positions: torch.Tensor, batch: int, seqlen: int
) -> torch.Tensor:
    fr = torch.view_as_real(freqs_cis)
    half = fr.shape[1]
    # expand to (B, S_max, 1, half, 2) then gather along the sequence dim
    fr_exp = fr.unsqueeze(0).unsqueeze(2).expand(batch, -1, -1, -1, -1)
    index = positions.view(batch, seqlen, 1, 1, 1).expand(batch, seqlen, 1, half, 2)
    fr_out = torch.gather(fr_exp, dim=1, index=index)
    return torch.view_as_complex(fr_out.contiguous())


def reshape_for_broadcast(
    freqs_cis: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    """Reshape the RoPE frequency tensor for broadcasting with ``x``.

    Args:
        freqs_cis: Precomputed complex frequency tensor of shape ``(max_seqlen, dim // 2)``.
        x: Target tensor; freq slice is aligned to ``x.shape[1]`` (seqlen) and ``x.shape[-1]``.
        positions: Position indices of shape ``(1, seqlen)`` or ``(B, seqlen)``.
            When ``None``, the first ``seqlen`` rows of ``freqs_cis`` are used directly.

    Returns:
        Frequency tensor broadcastable with ``x``.
    """
    ndim = x.ndim
    if ndim <= 1:
        raise ValueError(f"x.ndim must be > 1, but got {ndim}")
    seqlen = x.shape[1]
    if positions is not None and positions.size(0) > 1:
        bz = x.shape[0]
        if positions.shape != (bz, seqlen):
            raise ValueError(
                f"positions.shape must be {(bz, seqlen)} for batched positions, "
                f"but got {tuple(positions.shape)}"
            )
        return _gather_freqs_cis_by_positions_batched(freqs_cis, positions, bz, seqlen)
    if positions is None:
        freqs_cis = freqs_cis[:seqlen]
    else:
        if positions.shape != (1, seqlen):
            raise ValueError(
                f"positions.shape must be {(1, seqlen)} for shared positions, "
                f"but got {tuple(positions.shape)}"
            )
        fr = torch.view_as_real(freqs_cis)
        pos_idx = positions.squeeze(0)
        if hasattr(pos_idx, "to_local"):
            pos_idx = pos_idx.to_local()
        if hasattr(fr, "to_local"):
            from torch.distributed.tensor import DTensor

            device_mesh, placements = (
                fr.device_mesh,  # pyrefly: ignore [missing-attribute]
                fr.placements,  # pyrefly: ignore [missing-attribute]
            )
            freqs_local = torch.view_as_complex(fr.to_local()[pos_idx].contiguous())
            freqs_cis = DTensor.from_local(
                freqs_local, device_mesh, placements, run_check=False
            )
        else:
            freqs_cis = torch.view_as_complex(fr[pos_idx].contiguous())
    if freqs_cis.shape != (seqlen, x.shape[-1]):
        raise ValueError(
            f"freqs_cis.shape must be {(seqlen, x.shape[-1])}, "
            f"but got {tuple(freqs_cis.shape)}"
        )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)
