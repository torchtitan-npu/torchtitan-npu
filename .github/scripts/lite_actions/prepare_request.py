#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Trusted CPU-only workflow -> Matrix and source-SHA-bound Artifact.

GitHub prepares one immutable, SHA-bound case plan. The dispatcher owns training.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

TEST = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
# Only the upstream organization may publish CI plans or consume results.
REPO = "torchtitan-npu/torchtitan-npu"
if (source := os.environ.get("GITHUB_REPOSITORY", REPO)) != REPO:
    raise ValueError("untrusted GitHub CI repository")
PIPELINES = {"a3-8p", "a3-16p", "a5-64p"}


def parse_cases(value: str) -> list[dict]:
    if len(value.encode()) > 65535:
        raise ValueError("test_cases input too long")
    cases = json.loads(value)
    if not isinstance(cases, list) or not cases:
        raise ValueError("test_cases selector count invalid")
    for case in cases:
        if not isinstance(case, dict) or len(case) != 1 or (("test_id" in case) == ("suite" in case)):
            raise ValueError("each request selects exactly one test_id or suite")
        name = case.get("test_id", case.get("suite"))
        if not isinstance(name, str) or not TEST.fullmatch(name):
            raise ValueError("invalid registered test/suite")
    return cases


from tests.integration_tests.tools.lite_actions.contract import create_plan, normalize


def expand_cases(selectors, defaults):
    from tests.integration_tests.tools.lite_actions.entrypoint import catalog, select

    catalogue = catalog()
    expanded = []
    seen = set()
    for selector in selectors:
        for test in select(catalogue, **selector):
            if test.test_name in seen:
                raise ValueError("duplicate expanded Test Case")
            seen.add(test.test_name)
            expanded.append(normalize(test, defaults, len(expanded)))
    return expanded


def main():
    defaults = {k: os.environ["CI_DEFAULT_" + k.upper()] for k in ("worker", "target_pool", "docker")}
    cases = expand_cases(parse_cases(os.environ["CI_TEST_CASES"]), defaults)
    plan = create_plan(
        cases,
        run_id=int(os.environ["GITHUB_RUN_ID"]),
        attempt=int(os.environ["GITHUB_RUN_ATTEMPT"]),
        sha=os.environ["CI_SOURCE_SHA"],
        repo=os.environ["GITHUB_REPOSITORY"],
        workflow=os.environ["CI_WORKFLOW"],
        defaults=defaults,
    )
    out = Path(".ci-request")
    out.mkdir(exist_ok=True)
    (out / "ci-request.json").write_text(json.dumps(plan, ensure_ascii=False, separators=(",", ":")) + "\n")
    with open(os.environ["GITHUB_OUTPUT"], "a") as f:
        f.write("matrix=" + json.dumps({"include": cases}, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.write("plan_digest=" + plan["plan_digest"] + "\n")


if __name__ == "__main__":
    main()
