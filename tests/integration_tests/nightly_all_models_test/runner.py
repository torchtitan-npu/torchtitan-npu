# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared execution adapter for OverrideDefinitions-based 8P/16P/64P cases.

TorchTitan's run_tests()/GPUPool remains the 1-node 8P test runner.
Cross-node orchestration stays in Lite Actions; only the per-node recipe and
TensorBoard assertions live here, never SSH/resource scheduling.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from tests.integration_tests import OverrideDefinitions


def dispatcher_port_env() -> dict[str, str]:
    """Apply dispatcher-issued rendezvous ports after model recipe defaults."""
    overrides = {}
    for key, target in (("LITE_CI_MASTER_PORT", "MASTER_PORT"),
                        ("LITE_CI_HCCL_IF_BASE_PORT", "HCCL_IF_BASE_PORT")):
        raw = os.environ.get(key)
        if raw is not None:
            if not raw.isascii() or not raw.isdecimal() or not 1024 <= int(raw) <= 65535:
                raise ValueError(f"invalid dispatcher-issued {key}")
            overrides[target] = raw
    for key, target in (("LITE_CI_HCCL_HOST_PORT_RANGE", "HCCL_HOST_SOCKET_PORT_RANGE"),
                        ("LITE_CI_HCCL_NPU_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE")):
        raw = os.environ.get(key)
        if raw is not None:
            match = __import__("re").fullmatch(r"([1-9][0-9]{3,4})-([1-9][0-9]{3,4})", raw)
            if not match or not 1024 <= int(match[1]) <= int(match[2]) <= 65535:
                raise ValueError(f"invalid dispatcher-issued {key}")
            overrides[target] = raw
    return overrides


def run_single(test: OverrideDefinitions, *, output_dir: Path | None = None) -> None:
    from tests.integration_tests.run_tests import run_tests

    parser = argparse.ArgumentParser(description=test.test_descr)
    if output_dir is None:
        parser.add_argument("output_dir", type=Path)
        output_dir = parser.parse_args().output_dir
    output = output_dir
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        parser.error("output_dir must be empty")
    # run_tests.run_single_test merges OverrideDefinitions.env_vars after
    # os.environ, so trusted dispatcher ports must also override the test's
    # per-Case defaults (not just the shell environment).
    from dataclasses import replace
    test = replace(test, env_vars={**(test.env_vars or {}), **dispatcher_port_env()})
    run_tests(argparse.Namespace(output_dir=str(output), ngpu=test.ngpu,
                                 test_name=test.test_name, exclude=None),
              [test], parallel=True)


def run_distributed(test: OverrideDefinitions, *, phase: str,
                    output_dir: Path) -> None:
    parser = argparse.ArgumentParser(description=test.test_descr)
    if phase not in ("launch", "verify"):
        parser.error("phase must be launch or verify")
    run_root = output_dir / test.test_name / "test_run"
    if phase == "verify":
        from tests.integration_tests.loss_compare import extract_losses_from_tensorboard
        values = extract_losses_from_tensorboard(run_root, "tb_phase_0")
        expected = set(test.expected_steps[0]) if test.expected_steps else set()
        if set(values) != expected:
            raise RuntimeError(f"{test.test_name}: expected steps {sorted(expected)}, got {sorted(values)}")
        print(f"[MULTINODE_VERIFY] PASS nodes={test.nnodes} ngpu={test.ngpu} "
              f"steps={sorted(values)} loss={values}", flush=True)
        return

    ips = [x.strip() for x in os.environ.get("NODE_IPS", "").split(",")]
    if len(ips) != test.nnodes or any(not x for x in ips):
        parser.error(f"NODE_IPS requires exactly {test.nnodes} nonempty IPs")
    if os.environ.get("NGPU", str(test.ngpu)) != str(test.ngpu):
        parser.error(f"NGPU must be {test.ngpu} per node")
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        parser.error("output_dir must be empty")
    if len(test.override_args) != 1:
        raise ValueError("multi-node CI case requires exactly one CLI phase")
    command = ["bash", test.train_script, "--dump_folder", str(run_root),
               *(test.train_args or ()), *test.override_args[0],
               "--metrics.save_tb_folder=tb_phase_0"]
    print("[MULTINODE_LAUNCH] " + " ".join(command), flush=True)
    env = {**os.environ, **(test.env_vars or {}), **dispatcher_port_env()}
    raise SystemExit(subprocess.call(command, env=env))
