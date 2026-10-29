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


SMOKE_WORLD = {"dispatch_smoke_8p": 8, "dispatch_smoke_16p": 16}


def is_dispatch_smoke(test: OverrideDefinitions) -> bool:
    return test.test_name in SMOKE_WORLD


def run_dispatch_smoke(test: OverrideDefinitions, *, output_dir: Path) -> None:
    """Launch only a tiny HCCL DP collective; never load model/checkpoint assets."""
    import json
    import sys

    world = SMOKE_WORLD[test.test_name]
    nodes = int(os.environ.get("NNODES", str(test.nnodes)))
    node_ips = [item.strip() for item in os.environ.get("NODE_IPS", "").split(",") if item.strip()]
    if nodes != test.nnodes or len(node_ips) != nodes:
        raise RuntimeError(f"smoke topology mismatch: nodes={nodes}, ips={node_ips}")
    node_rank = 0
    if nodes > 1:
        import socket
        local_ips = set()
        for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            local_ips.add(item[4][0])
        try:
            import subprocess
            local_ips.update(subprocess.check_output(
                ["hostname", "-I"], text=True, timeout=3).split())
        except (OSError, subprocess.SubprocessError):
            pass
        matches = [index for index, ip in enumerate(node_ips) if ip in local_ips]
        if len(matches) != 1:
            raise RuntimeError(
                f"cannot map local host to exactly one NODE_IPS entry: "
                f"local_ips={sorted(local_ips)}, node_ips={node_ips}")
        node_rank = matches[0]
    if os.environ.get("NGPU", str(test.ngpu)) != str(test.ngpu):
        raise RuntimeError(f"smoke expects NGPU={test.ngpu}")
    if not 0 <= node_rank < nodes:
        raise RuntimeError(f"invalid NODE_RANK={node_rank} for {nodes} nodes")
    port = os.environ.get("LITE_CI_MASTER_PORT")
    if not port or not port.isdecimal() or not 1024 <= int(port) <= 65535:
        raise RuntimeError("missing/invalid dispatcher-issued LITE_CI_MASTER_PORT")

    env = os.environ.copy()
    env["LITE_SMOKE_EXPECTED_WORLD"] = str(world)
    env["MASTER_ADDR"] = node_ips[0]
    env["MASTER_PORT"] = port
    if "LITE_CI_HCCL_IF_BASE_PORT" in env:
        env["HCCL_IF_BASE_PORT"] = env["LITE_CI_HCCL_IF_BASE_PORT"]
    for source, target in (
        ("LITE_CI_HCCL_HOST_PORT_RANGE", "HCCL_HOST_SOCKET_PORT_RANGE"),
        ("LITE_CI_HCCL_NPU_PORT_RANGE", "HCCL_NPU_SOCKET_PORT_RANGE"),
    ):
        if source in env:
            env[target] = env[source]
    env["OMP_NUM_THREADS"] = "1"

    cmd = [sys.executable, "-m", "torch.distributed.run",
           "--nnodes", str(nodes), "--nproc-per-node", str(test.ngpu),
           "--node-rank", str(node_rank), "--master-addr", node_ips[0],
           "--master-port", port,
           "-m", "tests.integration_tests.tools.lite_actions.dispatch_smoke_worker"]
    print("[DISPATCH_SMOKE_LAUNCH] " + " ".join(cmd), flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise RuntimeError("smoke output_dir must be empty")
    import subprocess
    try:
        subprocess.run(cmd, env=env, check=True, timeout=75)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("dispatch smoke exceeded 75-second execution budget") from exc
    result = {"status": "PASS", "test_id": test.test_name,
              "world_size": world, "nodes": nodes, "devices_per_node": test.ngpu}
    (output_dir / "dispatch_smoke_result.json").write_text(
        json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    print("[DISPATCH_SMOKE_RESULT] " + json.dumps(result, sort_keys=True), flush=True)


def verify_dispatch_smoke(test: OverrideDefinitions, output_dir: Path) -> None:
    import json
    found = output_dir / "dispatch_smoke_result.json"
    if not found.is_file():
        raise RuntimeError(f"missing dispatch smoke result marker: {found}")
    result = json.loads(found.read_text(encoding="utf-8"))
    expected = SMOKE_WORLD[test.test_name]
    if result.get("status") != "PASS" or result.get("world_size") != expected:
        raise RuntimeError(f"invalid dispatch smoke result: {result}")
    print(f"[DISPATCH_SMOKE_VERIFY] PASS test={test.test_name} world_size={expected}", flush=True)


def run_single(test: OverrideDefinitions, *, output_dir: Path | None = None) -> None:
    if is_dispatch_smoke(test):
        if output_dir is None:
            parser = argparse.ArgumentParser(description=test.test_descr)
            parser.add_argument("output_dir", type=Path)
            output_dir = parser.parse_args().output_dir
        run_dispatch_smoke(test, output_dir=output_dir)
        return

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
        if is_dispatch_smoke(test):
            verify_dispatch_smoke(test, output_dir)
            return
        from tests.integration_tests.loss_compare import extract_losses_from_tensorboard
        values = extract_losses_from_tensorboard(run_root, "tb_phase_0")
        expected = set(test.expected_steps[0]) if test.expected_steps else set()
        if set(values) != expected:
            raise RuntimeError(f"{test.test_name}: expected steps {sorted(expected)}, got {sorted(values)}")
        print(f"[MULTINODE_VERIFY] PASS nodes={test.nnodes} ngpu={test.ngpu} "
              f"steps={sorted(values)} loss={values}", flush=True)
        return

    if is_dispatch_smoke(test):
        run_dispatch_smoke(test, output_dir=output_dir)
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
