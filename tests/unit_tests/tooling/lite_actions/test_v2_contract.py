"""CPU-only V2 plan, independent Matrix, waiter and workflow admission."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[4]
PREP = ROOT / ".github/scripts/lite_actions/prepare_request.py"
WAIT = ROOT / ".github/scripts/lite_actions/wait_result.py"


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prepare = load("v2_prepare", PREP)
waiter = load("v2_waiter", WAIT)


class V2Contract(unittest.TestCase):
    def test_v2_matrix_and_artifact_identical(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            try:
                os.chdir(td)
                env = {"CI_TEST_CASES": '[{"suite":"a3_8p_tests"}]',
                       "CI_PROTOCOL_VERSION": "2", "CI_MAX_SELECTORS": "2",
                       "CI_MAX_CASES": "4", "CI_PIPELINE": "a3-8p",
                       "GITHUB_REPOSITORY": "depeng1994/torchtitan-npu",
                       "GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "1",
                       "GITHUB_SHA": "a"*40,
                       "GITHUB_OUTPUT": str(Path(td) / "gh-output.txt")}
                with patch.dict(os.environ, env):
                    # This runs against real model-owned OverrideDefinitions.
                    prepare.main()
                request = json.loads(Path(".ci-request/ci-request.json").read_text())
                matrix = json.loads(Path("gh-output.txt").read_text().splitlines()[0].split("=",1)[1])
                self.assertEqual(request["schema"], 2)
                self.assertEqual(request["repo"], env["GITHUB_REPOSITORY"])
                self.assertEqual(matrix["include"][0]["test_id"],
                                 "dsv4_flash_a3_8p_example")
                self.assertEqual(len(matrix["include"]), 2)
                self.assertEqual([x["test_id"] for x in matrix["include"]],
                                 [x["test_id"] for x in request["cases"]])
                unsigned={k:v for k,v in request.items() if k!="plan_digest"}
                self.assertEqual(request["plan_digest"], prepare.digest_plan(unsigned))
            finally:
                os.chdir(cwd)

    def test_selector_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            prepare.expand_cases([{"suite": "a3_8p_tests"},
                                  {"test_id": "dsv4_flash_a3_8p_adamw"}], 4)

    def test_matrix_result_identity_and_failed_case_isolated(self):
        ref={"schema":2,"repo":"depeng1994/torchtitan-npu",
             "sha":"a"*40,"run_id":91,"attempt":1,"pipeline":"a3-8p",
             "test_id":"dsv4_flash_a3_8p_adamw","plan_digest":"b"*64,
             "status":"FAIL","exit_code":42,"last_20_lines":["OOM"]}
        comment={"user":{"login":"depeng1994"},
                 "body":"Lite-CI-V2-RESULT 91:1:dsv4_flash_a3_8p_adamw\n"+json.dumps(ref)}
        q=dict(sha="a"*40,run_id=91,attempt=1,pipeline="a3-8p",
               test_id="dsv4_flash_a3_8p_adamw",plan_digest="b"*64)
        self.assertEqual(waiter.find_case_result([comment],**q),ref)
        for change in ({"attempt":2},{"plan_digest":"c"*64},
                       {"test_id":"dsv4_flash_a3_8p_example"},{"sha":"c"*40}):
            self.assertIsNone(waiter.find_case_result([comment],**{**q,**change}))
        comment["user"]["login"]="attacker"
        self.assertIsNone(waiter.find_case_result([comment],**q))

    def test_partial_matrix_rerun_without_artifact_fails_fast(self):
        import io
        with patch.object(waiter.urllib.request, "urlopen",
                          return_value=io.BytesIO(
                              b'{"artifacts":[{"name":"lite-ci-request-1",'
                              b'"expired":false}]}')):
            with self.assertRaisesRegex(ValueError, "Partial Matrix reruns"):
                waiter.ensure_attempt_artifact("test-token",1234,2)
        with patch.object(waiter.urllib.request, "urlopen",
                          return_value=io.BytesIO(
                              b'{"artifacts":[{"name":"lite-ci-request-2",'
                              b'"expired":false}]}')):
            waiter.ensure_attempt_artifact("test-token",1234,2)

    def test_workflow_is_untrusted_pr_closed_and_matrix_native(self):
        for file in ("a3-8p-lite-actions.yml", "a3-16p-lite-actions.yml"):
            data=(ROOT/".github/workflows"/file).read_text()
            for expected in ("workflow_dispatch:", "refs/heads/master)",
                             "fromJSON(needs.prepare.outputs.matrix)", "fail-fast: false",
                             "CI_PROTOCOL_VERSION: \"2\"", "--test-id",
                             "actions/upload-artifact@v4"):
                self.assertIn(expected,data)
            self.assertNotIn("runs-on: self-hosted",data)

    def test_a5_workflow_is_validly_disabled_without_nonstandard_concurrency_keys(self):
        data=(ROOT/".github/workflows/a5-64p-lite-actions.yml").read_text()
        self.assertIn('CI_CHANNEL_ENABLED: "false"', data)
        self.assertIn("cancel-in-progress: false", data)
        self.assertNotIn("queue:", data)
        self.assertNotIn("runs-on: self-hosted", data)


if __name__ == "__main__":
    unittest.main()
