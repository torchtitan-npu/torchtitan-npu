# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from functools import wraps

from torchtitan.config import Comm as CommConfig
from torchtitan.distributed import utils as distributed_utils
from torchtitan.distributed.utils import init_distributed
from torchtitan.tools.logging import logger


@wraps(init_distributed)
def init_distributed_with_hccl_flight_recorder(
    comm_config: CommConfig,
    *args,
    **kwargs,
) -> int:
    def _warn_overwrite_env(env, val):
        if env in os.environ:
            logger.warning(
                f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config"
            )
        os.environ[env] = val

    # enable torch hccl flight recorder in the mode that would dump files if timeout is detected
    _warn_overwrite_env("TORCH_HCCL_TRACE_BUFFER_SIZE", str(comm_config.trace_buf_size))
    if comm_config.trace_buf_size > 0:
        # dump on timeout by default if trace buffer is enabled
        _warn_overwrite_env("HCCL_ASYNC_ERROR_HANDLING", "1")
        _warn_overwrite_env("TORCH_HCCL_ENABLE_MONITORING", "1")
        _warn_overwrite_env("TORCH_HCCL_DUMP_ON_TIMEOUT", "1")

        base_folder = kwargs.get("base_folder", "")
        dump_dir = os.path.join(base_folder, comm_config.save_traces_folder)
        prefix = comm_config.save_traces_file_prefix
        os.makedirs(dump_dir, exist_ok=True)
        _warn_overwrite_env("TORCH_HCCL_DEBUG_INFO_TEMP_FILE", f"{dump_dir}/{prefix}")

    return init_distributed(comm_config, *args, **kwargs)


distributed_utils.init_distributed = init_distributed_with_hccl_flight_recorder
