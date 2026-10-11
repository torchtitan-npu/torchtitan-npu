#!/usr/bin/env python3
"""Trusted CPU-only workflow_dispatch -> V2 Matrix and bound Artifact.

GitHub prepares one immutable, SHA-bound case plan. The dispatcher owns training.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

TEST = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPO = "depeng1994/torchtitan-npu"
PIPELINES = {"a3-8p", "a3-16p", "a5-64p"}


def parse_cases(value: str) -> list[dict]:
    if len(value.encode()) > 8192:
        raise ValueError("test_cases input too long")
    cases = json.loads(value)
    if not isinstance(cases, list) or not 1 <= len(cases) <= 12:
        raise ValueError("test_cases selector count invalid")
    for case in cases:
        if (not isinstance(case, dict) or len(case) != 1
                or (("test_id" in case) == ("suite" in case))):
            raise ValueError("each request selects exactly one test_id or suite")
        name = case.get("test_id", case.get("suite"))
        if not isinstance(name, str) or not TEST.fullmatch(name):
            raise ValueError("invalid registered test/suite")
    if len(cases) > int(os.environ.get("CI_MAX_SELECTORS", os.environ.get("CI_MAX_CASES", "2"))):
        raise ValueError("too many selectors")
    return cases


def expand_cases(selectors: list[dict], limit: int) -> list[dict]:
    """Use model-owned definitions; imports are trusted from validated master."""
    from tests.integration_tests.tools.lite_actions.entrypoint import catalog, select
    catalogue = catalog()
    expanded, seen = [], set()
    for selector in selectors:
        for test in select(catalogue, **selector):
            if test.test_name in seen:
                raise ValueError("duplicate expanded Test Case")
            if len(expanded) >= limit:
                raise ValueError("expanded Test Case count exceeds budget")
            if not isinstance(test.ngpu, int) or not isinstance(test.nnodes, int):
                raise ValueError("invalid resource requirements")
            if test.ngpu < 1 or test.nnodes < 1:
                raise ValueError("invalid resource requirements")
            label = test.test_descr.strip()
            if (not label or len(label) > 115 or any(ord(x) < 32 for x in label)):
                raise ValueError("invalid Test Case description")
            expanded.append({"test_id": test.test_name, "nnodes": test.nnodes,
                             "ngpu": test.ngpu, "label": label})
            seen.add(test.test_name)
    return expanded


def digest_plan(plan: dict) -> str:
    canon = json.dumps(plan, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def create_plan(*, cases: list[dict], run_id: int, attempt: int, sha: str,
                repo: str, pipeline: str) -> dict:
    if repo != REPO or pipeline not in PIPELINES:
        raise ValueError("untrusted repository/pipeline")
    if run_id < 1 or attempt < 1 or not SHA.fullmatch(sha):
        raise ValueError("invalid GitHub identity")
    plan = {"schema": 2, "repo": repo, "run_id": run_id, "attempt": attempt,
            "sha": sha, "pipeline": pipeline, "cases": cases}
    return {**plan, "plan_digest": digest_plan(plan)}


def main() -> None:
    selectors = parse_cases(os.environ["CI_TEST_CASES"])
    version = os.environ.get("CI_PROTOCOL_VERSION", "1")
    request_dir = Path(".ci-request")
    request_dir.mkdir(exist_ok=True)
    if version == "1":
        # Old workflow stays functional during the dispatcher-first rollout.
        if len(selectors) > int(os.environ.get("CI_MAX_CASES", "2")):
            raise ValueError("v1 case budget exceeded")
        result = {"schema": 1, "run_id": int(os.environ["GITHUB_RUN_ID"]),
                  "attempt": int(os.environ["GITHUB_RUN_ATTEMPT"]),
                  "sha": os.environ["GITHUB_SHA"], "cases": selectors}
        print("Prepared legacy request:", len(selectors), "selectors")
    elif version == "2":
        cases = expand_cases(selectors, int(os.environ.get("CI_MAX_CASES", "2")))
        result = create_plan(
            cases=cases, run_id=int(os.environ["GITHUB_RUN_ID"]),
            attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
            sha=os.environ["GITHUB_SHA"],
            repo=os.environ["GITHUB_REPOSITORY"],
            pipeline=os.environ["CI_PIPELINE"],
        )
        matrix = {"include": [{"test_id": case["test_id"], "label": case["label"]}
                              for case in cases]}
        output = os.environ.get("GITHUB_OUTPUT")
        if not output:
            raise ValueError("GITHUB_OUTPUT required for Matrix")
        with open(output, "a", encoding="utf-8") as handle:
            handle.write("matrix=" + json.dumps(matrix, ensure_ascii=False,
                                               separators=(",", ":")) + "\n")
            handle.write("plan_digest=" + result["plan_digest"] + "\n")
        print("Prepared V2 Matrix:", [case["test_id"] for case in cases])
    else:
        raise ValueError("unsupported CI protocol version")
    (request_dir / "ci-request.json").write_text(
        json.dumps(result, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
