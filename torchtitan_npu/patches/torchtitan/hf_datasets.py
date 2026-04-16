# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is derived from torchtitan,
# https://github.com/pytorch/torchtitan/blob/v0.2.2/torchtitan/hf_datasets/text_datasets.py
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset
from torchtitan.hf_datasets import text_datasets as text_datasets_utils
from torchtitan.hf_datasets.text_datasets import (
    _process_c4_text,
    build_text_dataloader,
    DatasetConfig,
    DATASETS,
)
from torchtitan.tools.logging import init_logger, logger

init_logger()

# Define new dataset configurations
new_datasets = {
    "enwiki-eod": DatasetConfig(
        path="tests/assets/enwiki",
        loader=lambda path: load_dataset(path, split="train"),
        sample_processor=_process_c4_text,
    ),
}

# Adding Datasets in Batches
added_datasets = []
for name, config in new_datasets.items():
    if name not in DATASETS:
        DATASETS[name] = config
        added_datasets.append(name)
        logger.info(
            f"[Dataset Patch] Successfully added dataset config: {name} (path: {config.path})"
        )
    else:
        logger.warning(f"[Dataset Patch] Dataset {name} already exists, skip adding")

# Summary print
if added_datasets:
    logger.info(
        f"[Dataset Patch] Added {len(added_datasets)} datasets in total: {added_datasets}"
    )
    logger.info(f"[Dataset Patch] All supported datasets now: {list(DATASETS.keys())}")
else:
    logger.info(
        f"[Dataset Patch] No new datasets to add, current supported: {list(DATASETS.keys())}"
    )


def mtp_build_text_dataloader(
    dp_world_size,
    dp_rank,
    tokenizer,
    job_config,
    infinite: bool = True,
):
    if (
        hasattr(job_config.training, "num_mtp_modules")
        and job_config.training.num_mtp_modules > 0
    ):
        if job_config.model.name in ["deepseek_v32"]:
            job_config.training.seq_len += job_config.training.num_mtp_modules
            result = build_text_dataloader(
                dp_world_size, dp_rank, tokenizer, job_config, infinite
            )
            job_config.training.seq_len -= job_config.training.num_mtp_modules
        else:
            raise AssertionError(
                "Multi Token Prediction Module only can be used for deepseek_v32 model now!"
            )
    else:
        result = build_text_dataloader(
            dp_world_size, dp_rank, tokenizer, job_config, infinite
        )
    return result


text_datasets_utils.build_text_dataloader = mtp_build_text_dataloader
