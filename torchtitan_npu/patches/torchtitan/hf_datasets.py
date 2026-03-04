# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datasets import load_dataset
from torchtitan.hf_datasets.text_datasets import DATASETS, DatasetConfig, _process_c4_text

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
        logger.info(f"[Dataset Patch] Successfully added dataset config: {name} (path: {config.path})")
    else:
        logger.warning(f"[Dataset Patch] Dataset {name} already exists, skip adding")

# Summary print
if added_datasets:
    logger.info(f"[Dataset Patch] Added {len(added_datasets)} datasets in total: {added_datasets}")
    logger.info(f"[Dataset Patch] All supported datasets now: {list(DATASETS.keys())}")
else:
    logger.info(f"[Dataset Patch] No new datasets to add, current supported: {list(DATASETS.keys())}")