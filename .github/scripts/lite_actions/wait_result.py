#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GitHub-hosted Waiter for local smoke and independent Matrix Case results."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import time
import urllib.parse
import urllib.request

# Only the upstream organization may publish CI plans or consume results.
REPO = "torchtitan-npu/torchtitan-npu"
if (source := os.environ.get("GITHUB_REPOSITORY", REPO)) != REPO:
    raise ValueError("untrusted GitHub CI repository")
# The dispatcher posts under its own identity, not necessarily the PR author.
# Organization repository variable is configured by the maintainer.
RESULT_AUTHOR = os.environ.get("CI_RESULT_AUTHOR") or os.environ.get("GITHUB_ACTOR", "")

PIPELINES = {
    "dispatcher-smoke": ("Dispatcher-Smoke", "npu-smi"),
    "a3-8p": ("A3-8p-CI-Example", "a3-8p"),
    "a3-16p": ("A3-16p-CI-Example", "a3-16p"),
    "a5-64p": ("A5-64p-CI", "a5-64p"),
}
TEST = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
STATUS = {"PASS", "FAIL", "NOT_RUN", "CANCELLED", "TIMED_OUT", "INFRA_ERROR", "RESOURCE_TIMEOUT"}


def find_smoke_result(comments, marker, sha, run_id, attempt, pipeline):
    """Validate the local smoke result keyed by run, attempt and SHA."""
    for item in reversed(comments):
        if not RESULT_AUTHOR or item.get("user", {}).get("login") != RESULT_AUTHOR:
            continue
        body = item.get("body", "")
        if not body.startswith(marker + "\n"):
            continue
        try:
            report = json.loads(body.split("\n", 1)[1])
        except (ValueError, IndexError):
            continue
        if (report.get("sha") == sha and report.get("run_id") == run_id
                and report.get("attempt") == attempt
                and report.get("pipeline") == pipeline
                and report.get("status") in ("PASS", "FAIL")):
            return report
    return None


def find_case_result(comments, *, sha: str, run_id: int, attempt: int,
                     pipeline: str, test_id: str, plan_digest: str) -> dict | None:
    marker = f"Lite-CI-RESULT {run_id}:{attempt}:{test_id}"
    for item in reversed(comments):
        if not RESULT_AUTHOR or item.get("user", {}).get("login") != RESULT_AUTHOR:
            continue
        body = item.get("body", "")
        if not body.startswith(marker + "\n"):
            continue
        try:
            report = json.loads(body.split("\n", 1)[1])
        except (ValueError, IndexError):
            continue
        if (report.get("schema") == 2 and report.get("repo") == REPO
                and report.get("sha") == sha
                and report.get("run_id") == run_id and report.get("attempt") == attempt
                and report.get("pipeline") == pipeline and report.get("test_id") == test_id
                and report.get("plan_digest") == plan_digest
                and report.get("status") in STATUS
                and isinstance(report.get("exit_code"), (int, type(None)))):
            return report
    return None



def find_progress(comments, *, sha: str, run_id: int, attempt: int,
                  pipeline: str, test_id: str, plan_digest: str = "") -> str:
    """Only the authenticated dispatcher can declare this Case RUNNING.

    Comments were fetched from the exact SHA endpoint; no JSON upload needed.
    """
    marker = f"LCI-R {run_id}:{attempt}:{test_id}"
    for item in comments:
        if (RESULT_AUTHOR and item.get("user", {}).get("login") == RESULT_AUTHOR
                and item.get("body") == marker):
            return "RUNNING"
    return "QUEUED"



BEIJING = dt.timezone(dt.timedelta(hours=8))


def heartbeat_line(*, status: str, current_case: str,
                   when: dt.datetime | None = None) -> str:
    """Human-readable Beijing-time progress; never sent by the dispatcher."""
    timestamp = (when or dt.datetime.now(dt.timezone.utc)).astimezone(BEIJING)
    return f"{timestamp:%Y/%m/%d %H:%M:%S} {current_case} CI {status.lower()}."



def get_comments(token: str, sha: str, since: str):
    # Fail closed on unexpected pagination; bounded API usage.
    items = []
    for page in range(1, 21):
        query = urllib.parse.urlencode({"per_page": 100, "since": since, "page": page})
        url = f"https://api.github.com/repos/{REPO}/commits/{sha}/comments?{query}"
        req = urllib.request.Request(url, headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "lite-actions-waiter/3",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        with urllib.request.urlopen(req, timeout=25) as response:
            page_items = json.load(response)
        items.extend(page_items)
        if len(page_items) < 100:
            return items
    raise RuntimeError("too many matching commit comments; cannot safely correlate")



def ensure_attempt_artifact(token: str, run_id: int, attempt: int) -> None:
    """Fail fast if a partial Matrix rerun lacks this attempt's trusted plan."""
    name = f"lite-ci-request-{attempt}"
    query = urllib.parse.urlencode({"name": name, "per_page": 10})
    url = f"https://api.github.com/repos/{REPO}/actions/runs/{run_id}/artifacts?{query}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "lite-actions-waiter/3",
        "X-GitHub-Api-Version": "2022-11-28",
    })
    for index in range(3):
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                listing = json.load(response)
            matches = [a for a in listing.get("artifacts", [])
                       if a.get("name") == name and not a.get("expired")]
            if len(matches) != 1:
                raise ValueError(
                    "No CI plan Artifact for this attempt. Partial Matrix reruns "
                    "are not supported; launch a NEW full workflow_dispatch run."
                )
            return
        except ValueError:
            raise
        except Exception:
            if index == 2:
                raise
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=PIPELINES, required=True)
    parser.add_argument("--test-id", help="Matrix Test ID (not used by dispatcher smoke)")
    args = parser.parse_args()
    display, pipeline = PIPELINES[args.case]
    if args.case == "dispatcher-smoke" and args.test_id:
        parser.error("dispatcher smoke must not specify --test-id")
    if args.case != "dispatcher-smoke" and not args.test_id:
        parser.error("Matrix CI requires --test-id")
    run_id = int(os.environ["GITHUB_RUN_ID"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    sha = os.environ.get("CI_SOURCE_SHA") or os.environ["GITHUB_SHA"]
    token = os.environ["GH_TOKEN"]
    digest = os.environ.get("CI_PLAN_DIGEST", "")
    if args.test_id:
        if not TEST.fullmatch(args.test_id) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            parser.error("invalid Matrix case/digest")
    if args.test_id:
        ensure_attempt_artifact(token, run_id, attempt)
    since = (dt.datetime.now(dt.timezone.utc) -
             dt.timedelta(days=2)).isoformat().replace("+00:00", "Z")
    started = time.monotonic()
    deadline = started + int(os.environ.get("CI_WAIT_TIMEOUT_SECONDS", "7400"))
    heartbeat_seconds = 60
    next_heartbeat = started
    progress = "QUEUED"
    current_case = args.test_id or "npu-smi"
    print(f"Waiting for {display} run={run_id} attempt={attempt} SHA={sha[:12]}"
          + f" current_case={current_case}", flush=True)
    while time.monotonic() < deadline:
        try:
            comments = get_comments(token, sha, since)
            progress = find_progress(
                comments, sha=sha, run_id=run_id, attempt=attempt,
                pipeline=pipeline, test_id=current_case,
                plan_digest=digest if args.test_id else "")
            if args.test_id:
                result = find_case_result(
                    comments, sha=sha, run_id=run_id, attempt=attempt,
                    pipeline=pipeline, test_id=args.test_id, plan_digest=digest)
                if result is not None:
                    status, code = result["status"], result.get("exit_code")
                    print(heartbeat_line(status=status, current_case=args.test_id))
                    print(f"CASE {args.test_id}: {status} rc={code}")
                    for line in result.get("last_20_lines", [])[-20:]:
                        print(line)
                    return 0 if status == "PASS" and code == 0 else 1
            else:
                marker = f"{display}-RESULT {run_id}:{attempt}"
                result = find_smoke_result(comments, marker, sha, run_id, attempt, pipeline)
                if result is not None:
                    print(heartbeat_line(status=result["status"], current_case=current_case))
                    print(f"RESULT {result['status']} exit_code={result['exit_code']}")
                    for entry in result.get("cases", []):
                        print(f"  {entry['test_name']}: {entry['status']} rc={entry['exit_code']}")
                    for line in result.get("last_20_lines", [])[-20:]:
                        print(line)
                    return 0 if result["status"] == "PASS" and result["exit_code"] == 0 else 1
        except Exception as exc:
            print(f"GitHub read retry: {type(exc).__name__}", flush=True)
        now = time.monotonic()
        if now >= next_heartbeat:
            print(heartbeat_line(status=progress, current_case=current_case),
                  flush=True)
            next_heartbeat = now + heartbeat_seconds
        time.sleep(min(12, max(0.1, next_heartbeat - time.monotonic()),
                       max(0.1, deadline - time.monotonic())))
    print("ERROR: timed out waiting for internal CI", flush=True)
    return 124


if __name__ == "__main__":
    raise SystemExit(main())
