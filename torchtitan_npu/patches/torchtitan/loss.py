# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/components/loss.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import functools

import torch
from torchtitan.components import loss as loss_utils
from torchtitan.components.loss import cross_entropy_loss
from torchtitan.config import JobConfig
from torchtitan.tools.logging import logger

from torchtitan_npu.config.custom_config import JobConfig as NpuJobConfig


def multi_token_cross_entropy_loss(
    preds: list[torch.Tensor],
    labels: torch.Tensor,
    job_config: JobConfig,
) -> torch.Tensor:

    main_loss = cross_entropy_loss(preds[0], labels[:, : job_config.training.seq_len])
    mtp_loss = 0

    for label_offset, pred in enumerate(preds[1:], 1):
        end_idx = label_offset + job_config.training.seq_len
        loss = cross_entropy_loss(
            pred,
            labels[:, label_offset:end_idx],
        )
        loss = (
            loss
            / job_config.training.num_mtp_modules  # pyrefly: ignore [missing-attribute]
        )
        mtp_loss = mtp_loss + loss
    return (
        main_loss
        + mtp_loss
        * job_config.training.mtp_loss_weight  # pyrefly: ignore [missing-attribute]
    )


def mtp_build_cross_entropy_loss(job_config: JobConfig, **kwargs):
    del kwargs  # delete any unused arguments
    if (
        isinstance(job_config, NpuJobConfig) and job_config.training.num_mtp_modules > 0
    ):  # pyrefly: ignore [missing-attribute]
        loss_fn = functools.partial(
            multi_token_cross_entropy_loss,
            job_config=job_config,
        )
        logger.info("Applying loss = main_loss + mtp_loss to the model")
    else:
        loss_fn = cross_entropy_loss
    if job_config.compile.enable and "loss" in job_config.compile.components:
        logger.info("Compiling the loss function with torch.compile")
        loss_fn = torch.compile(loss_fn, backend=job_config.compile.backend)
    return loss_fn


loss_utils.build_cross_entropy_loss = mtp_build_cross_entropy_loss
