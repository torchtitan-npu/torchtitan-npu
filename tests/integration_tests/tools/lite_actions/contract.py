# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure schema-3 contract, mirrored byte-for-byte in the two repositories."""

import hashlib
import json
import re

REPO = "torchtitan-npu/torchtitan-npu"
WORKFLOWS = {"nightly_fulltests.yml", "lite_actions_dispatcher_smoke.yml"}
WORKERS = {"blue", "yellow"}
TEST = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
KINDS = {"train", "npu_smoke", "worker_probe"}
STATUS = {
    "PASS",
    "FAIL",
    "INFRA_ERROR",
    "RESOURCE_TIMEOUT",
    "TIMED_OUT",
    "CANCELLED",
    "NOT_RUN",
}
CASE_KEYS = {
    "case_id",
    "order",
    "test_id",
    "label",
    "kind",
    "worker",
    "target_pool",
    "resources",
    "docker",
    "execution_timeout_seconds",
}


def digest(plan):
    return hashlib.sha256(
        json.dumps(
            plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def normalize(test, defaults, order):
    override = getattr(test, "ci_overrides", None) or {}
    if not isinstance(override, dict) or set(override) - {
        "worker",
        "target_pool",
        "docker",
    }:
        raise ValueError("unknown case override")
    config = dict(defaults)
    for k, v in override.items():
        if (
            not isinstance(v, str)
            or not v
            or len(v) > 256
            or any(ord(c) < 32 for c in v)
        ):
            raise ValueError("invalid case override")
        config[k] = v
    kind = getattr(test, "ci_kind", "train")
    if kind not in KINDS or config.get("worker") not in WORKERS:
        raise ValueError("invalid case kind/worker")
    if not TEST.fullmatch(test.test_name):
        raise ValueError("invalid test ID")
    if kind == "worker_probe":
        if set(override) & {"target_pool", "docker"}:
            raise ValueError("probe cannot request resources")
        pool = docker = None
        nodes = ngpu = 0
    else:
        pool, docker = config.get("target_pool"), config.get("docker")
        if not isinstance(pool, str) or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", pool):
            raise ValueError("invalid pool")
        if (
            not isinstance(docker, str)
            or not docker
            or len(docker) > 256
            or any(ord(c) < 33 for c in docker)
        ):
            raise ValueError("invalid docker")
        nodes, ngpu = test.nnodes, test.ngpu
        if (
            type(nodes) is not int
            or type(ngpu) is not int
            or not 1 <= nodes <= 16
            or not 1 <= ngpu <= 256
        ):
            raise ValueError("invalid topology")
    label = test.test_descr.strip()
    if not label or len(label) > 115 or any(ord(c) < 32 for c in label):
        raise ValueError("invalid label")
    timeout = test.timeout or (90 if kind == "worker_probe" else 7200)
    if type(timeout) is not int or not 1 <= timeout <= 14400:
        raise ValueError("invalid execution timeout")
    return dict(
        case_id=f"case_{order:03d}_{test.test_name}",
        order=order,
        test_id=test.test_name,
        label=label,
        kind=kind,
        worker=config["worker"],
        target_pool=pool,
        resources=dict(nodes=nodes, devices_per_node=ngpu),
        docker=docker,
        execution_timeout_seconds=timeout,
    )


def create_plan(cases, *, run_id, attempt, sha, workflow, defaults, repo=REPO):
    plan = dict(
        schema=3,
        repo=repo,
        workflow=workflow,
        run_id=run_id,
        attempt=attempt,
        sha=sha,
        defaults=defaults,
        cases=cases,
    )
    plan["plan_digest"] = digest(plan)
    parse_plan(plan, dict(id=run_id, run_attempt=attempt, head_sha=sha), workflow)
    return plan


def parse_plan(plan, run, workflow):
    if not isinstance(plan, dict) or set(plan) != {
        "schema",
        "repo",
        "workflow",
        "run_id",
        "attempt",
        "sha",
        "defaults",
        "cases",
        "plan_digest",
    }:
        raise ValueError("invalid plan structure")
    if (
        plan["schema"] != 3
        or plan["repo"] != REPO
        or workflow not in WORKFLOWS
        or plan["workflow"] != workflow
        or type(plan["run_id"]) is not int
        or plan["run_id"] < 1
        or type(plan["attempt"]) is not int
        or plan["attempt"] < 1
        or plan["run_id"] != run["id"]
        or plan["attempt"] != run.get("run_attempt", 1)
        or plan["sha"] != run["head_sha"]
        or not re.fullmatch(r"[0-9a-f]{40}", str(plan["sha"]))
    ):
        raise ValueError("plan identity mismatch")
    if (
        digest({k: v for k, v in plan.items() if k != "plan_digest"})
        != plan["plan_digest"]
    ):
        raise ValueError("plan digest mismatch")
    defaults = plan["defaults"]
    if (
        not isinstance(defaults, dict)
        or set(defaults) != {"worker", "target_pool", "docker"}
        or any(not isinstance(x, str) for x in defaults.values())
    ):
        raise ValueError("invalid defaults")
    items = plan["cases"]
    if not isinstance(items, list) or not 1 <= len(items) <= 256:
        raise ValueError("invalid case count")
    seen = set()
    for i, c in enumerate(items):
        if not isinstance(c, dict) or set(c) != CASE_KEYS:
            raise ValueError("invalid case structure")
        if (
            not isinstance(c["test_id"], str)
            or not TEST.fullmatch(c["test_id"])
            or c["test_id"] in seen
        ):
            raise ValueError("duplicate/invalid test")
        seen.add(c["test_id"])
        if (
            c["order"] != i
            or type(c["order"]) is not int
            or c["case_id"] != f"case_{i:03d}_{c['test_id']}"
        ):
            raise ValueError("invalid case order/ID")
        from types import SimpleNamespace

        expected = normalize(
            SimpleNamespace(
                test_name=c["test_id"],
                test_descr=c["label"],
                nnodes=c["resources"].get("nodes"),
                ngpu=c["resources"].get("devices_per_node"),
                ci_kind=c["kind"],
                ci_overrides=(
                    {"worker": c["worker"]}
                    if c["kind"] == "worker_probe"
                    else {
                        "worker": c["worker"],
                        "target_pool": c["target_pool"],
                        "docker": c["docker"],
                    }
                ),
                timeout=c["execution_timeout_seconds"],
            ),
            defaults,
            i,
        )
        if expected != c:
            raise ValueError("noncanonical case")
    return items
