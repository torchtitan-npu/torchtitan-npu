#!/usr/bin/env python3
"""GitHub-hosted Waiter: V1 aggregate or V2 independent Matrix Case checks."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import re
import time
import urllib.parse
import urllib.request

REPO = "depeng1994/torchtitan-npu"
PIPELINES = {
    "a3-smoke": ("Dispatcher-Smoke", "npu-smi"),
    "a3-8p": ("A3-8p-CI-Example", "a3-8p"),
    "a3-16p": ("A3-16p-CI-Example", "a3-16p"),
    "a5-64p": ("A5-64p-CI", "a5-64p"),
}
TEST = re.compile(r"[a-z][a-z0-9_]{0,79}\Z")
STATUS = {"PASS", "FAIL", "NOT_RUN", "CANCELLED", "TIMED_OUT", "INFRA_ERROR", "RESOURCE_TIMEOUT"}


def find_result(comments, marker, sha, run_id, attempt, pipeline):
    """Legacy V1: retained until all in-flight V1 workflows drain."""
    for item in reversed(comments):
        if item.get("user", {}).get("login") != "depeng1994":
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
    marker = f"Lite-CI-V2-RESULT {run_id}:{attempt}:{test_id}"
    for item in reversed(comments):
        if item.get("user", {}).get("login") != "depeng1994":
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


def get_comments(token: str, sha: str, since: str):
    # Fail closed on unexpected pagination; bounded API usage.
    items = []
    for page in range(1, 5):
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
                    "No V2 plan Artifact for this attempt. Partial Matrix reruns "
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
    parser.add_argument("--test-id", help="V2 Matrix Test ID, omit for V1/smoke")
    args = parser.parse_args()
    display, pipeline = PIPELINES[args.case]
    run_id = int(os.environ["GITHUB_RUN_ID"])
    attempt = int(os.environ["GITHUB_RUN_ATTEMPT"])
    sha = os.environ["GITHUB_SHA"]
    token = os.environ["GH_TOKEN"]
    digest = os.environ.get("CI_PLAN_DIGEST", "")
    if args.test_id:
        if not TEST.fullmatch(args.test_id) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            parser.error("invalid V2 Matrix case/digest")
    if args.test_id:
        ensure_attempt_artifact(token, run_id, attempt)
    since = (dt.datetime.now(dt.timezone.utc) -
             dt.timedelta(days=2)).isoformat().replace("+00:00", "Z")
    deadline = time.monotonic() + int(os.environ.get("CI_WAIT_TIMEOUT_SECONDS", "7400"))
    print(f"Waiting for {display} run={run_id} attempt={attempt} SHA={sha[:12]}"
          + (f" test={args.test_id}" if args.test_id else ""), flush=True)
    while time.monotonic() < deadline:
        try:
            comments = get_comments(token, sha, since)
            if args.test_id:
                result = find_case_result(
                    comments, sha=sha, run_id=run_id, attempt=attempt,
                    pipeline=pipeline, test_id=args.test_id, plan_digest=digest)
                if result is not None:
                    status, code = result["status"], result.get("exit_code")
                    print(f"CASE {args.test_id}: {status} rc={code}")
                    for line in result.get("last_20_lines", [])[-20:]:
                        print(line)
                    return 0 if status == "PASS" and code == 0 else 1
            else:
                marker = f"{display}-RESULT {run_id}:{attempt}"
                result = find_result(comments, marker, sha, run_id, attempt, pipeline)
                if result is not None:
                    print(f"RESULT {result['status']} exit_code={result['exit_code']}")
                    for entry in result.get("cases", []):
                        print(f"  {entry['test_name']}: {entry['status']} rc={entry['exit_code']}")
                    for line in result.get("last_20_lines", [])[-20:]:
                        print(line)
                    return 0 if result["status"] == "PASS" and result["exit_code"] == 0 else 1
        except Exception as exc:
            print(f"GitHub read retry: {type(exc).__name__}", flush=True)
        time.sleep(12)
    print("ERROR: timed out waiting for internal CI", flush=True)
    return 124


if __name__ == "__main__":
    raise SystemExit(main())
