# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schema-3 producer, waiter correlation, workflow and output boundary checks."""

import copy
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import yaml
from tests.integration_tests.tools.lite_actions.contract import create_plan

ROOT = Path(__file__).resolve().parents[4]
D = dict(worker="blue", target_pool="a3_shared_blue", docker="dsv4.1_train_torchtitan_a3:cann_master_newest_0916")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class CIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prep = load("prepare", ROOT / ".github/scripts/lite_actions/prepare_request.py")
        cls.waiter = load("waiter", ROOT / ".github/scripts/lite_actions/wait_result.py")

    def cases(self):
        return self.prep.expand_cases([dict(suite="a3_8p_tests")], D)

    def test_real_dsv4_defaults_and_topology(self):
        cases = self.cases()
        self.assertEqual([c["test_id"] for c in cases], ["dsv4_flash_a3_8p_example", "dsv4_flash_a3_8p_adamw"])
        self.assertTrue(
            all(c["resources"] == dict(nodes=1, devices_per_node=8) and c["docker"] == D["docker"] for c in cases)
        )

    def test_selector_overlap_rejected(self):
        with self.assertRaises(ValueError):
            self.prep.expand_cases([dict(suite="a3_8p_tests"), dict(test_id="dsv4_flash_a3_8p_example")], D)

    def test_mixed_smoke_workers_and_topologies(self):
        cases = self.prep.expand_cases(
            [dict(suite=s) for s in ["worker_probe_tests", "a3_8p_smoke_tests", "a3_16p_smoke_tests"]], D
        )
        self.assertEqual(len(cases), 4)
        self.assertEqual([c["worker"] for c in cases], ["blue", "yellow", "blue", "blue"])
        self.assertEqual([c["resources"]["nodes"] for c in cases], [0, 0, 1, 2])

    def test_matrix_and_artifact_identical_at_pinned_head(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            try:
                os.chdir(td)
                env = dict(
                    CI_TEST_CASES='[{"suite":"a3_8p_tests"}]',
                    CI_WORKFLOW="nightly_fulltests.yml",
                    CI_SOURCE_SHA="a" * 40,
                    GITHUB_REPOSITORY="torchtitan-npu/torchtitan-npu",
                    GITHUB_RUN_ID="7",
                    GITHUB_RUN_ATTEMPT="2",
                    GITHUB_OUTPUT=td + "/output",
                    **{"CI_DEFAULT_" + k.upper(): v for k, v in D.items()},
                )
                with patch.dict(os.environ, env):
                    self.prep.main()
                plan = json.loads(Path(".ci-request/ci-request.json").read_text())
                outputs = dict(s.split("=", 1) for s in Path("output").read_text().splitlines())
                self.assertEqual(json.loads(outputs["matrix"])["include"], plan["cases"])
                self.assertEqual(plan["sha"], "a" * 40)
                self.assertEqual(plan["schema"], 3)
            finally:
                os.chdir(cwd)

    def result_fixture(self):
        item = self.cases()[0]
        expected = dict(
            schema=3,
            repo="torchtitan-npu/torchtitan-npu",
            workflow="nightly_fulltests.yml",
            run_id=7,
            attempt=2,
            sha="a" * 40,
            plan_digest="b" * 64,
            case_id=item["case_id"],
            test_id=item["test_id"],
            worker=item["worker"],
            target_pool=item["target_pool"],
            docker=item["docker"],
            resources=item["resources"],
            status="PASS",
            exit_code=0,
            cleanup_verified=True,
            last_20_lines=["step: 5 loss: 1"],
        )
        args = dict(
            run_id=7,
            attempt=2,
            sha="a" * 40,
            digest="b" * 64,
            item=item,
            workflow="nightly_fulltests.yml",
            author="reporter",
        )

        def comment(result, author="reporter"):
            return dict(
                user=dict(login=author),
                body=f"Lite-CI-V3 7:2:{item['case_id']}\n" + json.dumps(result, separators=(",", ":")),
            )

        return expected, args, comment

    def test_waiter_accepts_exact_result(self):
        result, args, comment = self.result_fixture()
        self.assertEqual(self.waiter.find_result([comment(result)], **args), result)

    def test_waiter_rejects_wrong_worker_author_attempt_sha_digest_and_case(self):
        result, args, comment = self.result_fixture()
        for key, value in [
            ("worker", "yellow"),
            ("attempt", 3),
            ("sha", "c" * 40),
            ("plan_digest", "d" * 64),
            ("case_id", "other"),
            ("cleanup_verified", False),
            ("exit_code", 1),
        ]:
            with self.subTest(key=key):
                self.assertIsNone(self.waiter.find_result([comment({**result, key: value})], **args))
        self.assertIsNone(self.waiter.find_result([comment(result, "stranger")], **args))

    def test_waiter_rejects_more_than_twenty_lines_or_control_characters(self):
        result, args, comment = self.result_fixture()
        for lines in [["x"] * 21, ["x" * 181], ["line\nextra"]]:
            with self.subTest(lines=lines):
                self.assertIsNone(self.waiter.find_result([comment({**result, "last_20_lines": lines})], **args))

    def test_only_two_business_workflows_and_no_schedule(self):
        callers = []
        for p in (ROOT / ".github/workflows").glob("*.yml"):
            data = yaml.load(p.read_text(), Loader=yaml.BaseLoader)
            for job in data.get("jobs", {}).values():
                if job.get("uses") == "./.github/workflows/lite_actions_common.yml":
                    callers.append(p.name)
        self.assertEqual(set(callers), {"nightly_fulltests.yml", "lite_actions_dispatcher_smoke.yml"})
        for name in callers:
            data = yaml.load((ROOT / ".github/workflows" / name).read_text(), Loader=yaml.BaseLoader)
            self.assertEqual(set(data["on"]), {"pull_request", "workflow_dispatch"})
            self.assertIn("result", data["jobs"])

    def test_common_keeps_exact_sha_matrix_and_bounded_waiter(self):
        p = ROOT / ".github/workflows/lite_actions_common.yml"
        text = p.read_text()
        data = yaml.load(text, Loader=yaml.BaseLoader)
        self.assertEqual(data["jobs"]["case"]["strategy"]["fail-fast"], "false")
        self.assertIn("github.event.pull_request.head.sha", text)
        self.assertNotIn("pull_request_target", text)

    def test_partial_rerun_requires_current_attempt_artifact(self):
        result, args, _ = self.result_fixture()
        env = dict(
            GITHUB_REPOSITORY="torchtitan-npu/torchtitan-npu",
            CI_RESULT_AUTHOR="reporter",
            CI_CASE=json.dumps(args["item"]),
            GITHUB_RUN_ID="7",
            GITHUB_RUN_ATTEMPT="3",
            GH_TOKEN="not-real",
            CI_SOURCE_SHA="a" * 40,
            CI_PLAN_DIGEST="b" * 64,
            CI_WORKFLOW="nightly_fulltests.yml",
        )
        with patch.dict(os.environ, env), patch.object(self.waiter, "get", return_value={"artifacts": []}):
            with self.assertRaisesRegex(ValueError, "re-run all jobs"):
                self.waiter.main()

    def test_waiter_skips_nonobject_json_and_still_finds_valid_result(self):
        result, args, comment = self.result_fixture()
        for invalid in [None, [], 0, {**result, "status": []}]:
            self.assertEqual(self.waiter.find_result([comment(result), comment(invalid)], **args), result)

    def test_waiter_rejects_utf8_and_escaped_request_over_budget(self):
        result, args, comment = self.result_fixture()
        for lines in [["中文" * 90] * 20, ['\\"' * 90] * 20]:
            self.assertIsNone(self.waiter.find_result([comment({**result, "last_20_lines": lines})], **args))

    def test_waiter_uses_etag_and_reuses_304_response(self):
        import io
        import urllib.error

        module = self.waiter
        module.HTTP_CACHE.clear()

        class Response:
            headers = {"ETag": "test-etag"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                return b"[]"

        with patch.object(module.urllib.request, "urlopen", return_value=Response()):
            self.assertEqual(module.get("/test", "isolated-token"), [])
        with patch.object(
            module.urllib.request,
            "urlopen",
            side_effect=urllib.error.HTTPError("https://example.invalid", 304, "unchanged", {}, io.BytesIO()),
        ) as open_url:
            self.assertEqual(module.get("/test", "isolated-token"), [])
            self.assertEqual(open_url.call_args.args[0].get_header("If-none-match"), "test-etag")
        module.HTTP_CACHE.clear()

    def test_waiter_treats_rate_limit_as_retryable(self):
        import io
        import urllib.error

        module = self.waiter
        error = urllib.error.HTTPError(
            "https://example.invalid", 403, "rate limit", {"X-RateLimit-Remaining": "0"}, io.BytesIO()
        )
        with patch.object(module.urllib.request, "urlopen", side_effect=error):
            with self.assertRaises(module.RetryableAPIError):
                module.get("/rate-test", "isolated-token")
