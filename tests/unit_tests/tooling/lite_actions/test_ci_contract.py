# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only CI plan, independent Matrix, waiter and workflow admission."""
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


class CIContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.prepare = load("ci_prepare", PREP)
        with patch.dict(os.environ, {"GITHUB_ACTOR": "ci-maintainer"}):
            cls.waiter = load("ci_waiter", WAIT)

    def test_matrix_and_artifact_identical(self):
        with tempfile.TemporaryDirectory() as td:
            cwd = Path.cwd()
            try:
                os.chdir(td)
                env = {"CI_TEST_CASES": '[{"suite":"a3_8p_tests"}]',
                       "CI_PIPELINE": "a3-8p",
                       "GITHUB_REPOSITORY": "torchtitan-npu/torchtitan-npu",
                       "GITHUB_RUN_ID": "12345", "GITHUB_RUN_ATTEMPT": "1",
                       "GITHUB_SHA": "a"*40,
                       "GITHUB_OUTPUT": str(Path(td) / "gh-output.txt")}
                with patch.dict(os.environ, env):
                    # This runs against real model-owned OverrideDefinitions.
                    self.prepare.main()
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
                self.assertEqual(request["plan_digest"], self.prepare.digest_plan(unsigned))
            finally:
                os.chdir(cwd)

    def test_expanded_suite_has_no_configured_budget(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        module = 'tests.integration_tests.tools.lite_actions.entrypoint'
        cases = [SimpleNamespace(test_name=f'case_{i:03d}', nnodes=1, ngpu=8,
                   test_descr=f'Test case {i}') for i in range(15)]
        with patch(f'{module}.catalog', return_value={'synthetic_tests': cases}):
            expanded = self.prepare.expand_cases([{'suite': 'synthetic_tests'}])
        self.assertEqual(len(expanded), 15)

    def test_ci_host_and_npu_hccl_ranges_validate_and_map(self):
        from tests.integration_tests.nightly_all_models_test.runner import dispatcher_port_env
        with patch.dict(os.environ, {"LITE_CI_HCCL_HOST_PORT_RANGE":"24000-24063",
                                  "LITE_CI_HCCL_NPU_PORT_RANGE":"28000-28063"}):
            x=dispatcher_port_env()
            self.assertEqual(x['HCCL_HOST_SOCKET_PORT_RANGE'],'24000-24063')
            self.assertEqual(x['HCCL_NPU_SOCKET_PORT_RANGE'],'28000-28063')
        for invalid in ('auto','16666','1-23','99999-100000','32000-31999'):
            with patch.dict(os.environ,{'LITE_CI_HCCL_NPU_PORT_RANGE':invalid}):
                with self.assertRaises(ValueError): dispatcher_port_env()

    def test_ci_port_override_wins_over_multinode_recipe_defaults(self):
        from unittest.mock import patch
        from tests.integration_tests.nightly_all_models_test.runner import dispatcher_port_env
        with patch.dict("os.environ", {"LITE_CI_MASTER_PORT": "16007",
                                    "LITE_CI_HCCL_IF_BASE_PORT": "20448"}):
            self.assertEqual(dispatcher_port_env(),
                             {"MASTER_PORT": "16007", "HCCL_IF_BASE_PORT": "20448"})
        for invalid in ("-1", "0", "65536", "not-a-port"):
            with patch.dict("os.environ", {"LITE_CI_MASTER_PORT": invalid}):
                with self.assertRaises(ValueError):
                    dispatcher_port_env()

    def test_selector_overlap_fails(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.prepare.expand_cases([{"suite": "a3_8p_tests"},
                                  {"test_id": "dsv4_flash_a3_8p_adamw"}])

    def test_matrix_result_identity_and_failed_case_isolated(self):
        ref={"schema":2,"repo":"torchtitan-npu/torchtitan-npu",
             "sha":"a"*40,"run_id":91,"attempt":1,"pipeline":"a3-8p",
             "test_id":"dsv4_flash_a3_8p_adamw","plan_digest":"b"*64,
             "status":"FAIL","exit_code":42,"last_20_lines":["OOM"]}
        comment={"user":{"login":"ci-maintainer"},
                 "body":"Lite-CI-RESULT 91:1:dsv4_flash_a3_8p_adamw\n"+json.dumps(ref)}
        q=dict(sha="a"*40,run_id=91,attempt=1,pipeline="a3-8p",
               test_id="dsv4_flash_a3_8p_adamw",plan_digest="b"*64)
        self.assertEqual(self.waiter.find_case_result([comment],**q),ref)
        for change in ({"attempt":2},{"plan_digest":"c"*64},
                       {"test_id":"dsv4_flash_a3_8p_example"},{"sha":"c"*40}):
            self.assertIsNone(self.waiter.find_case_result([comment],**{**q,**change}))
        comment["user"]["login"]="attacker"
        self.assertIsNone(self.waiter.find_case_result([comment],**q))

    def test_waiter_progress_heartbeat_checks_author_and_identity(self):
        args = dict(sha="a"*40,run_id=123,attempt=2,pipeline="a3-8p",
                    test_id="dsv4_flash_a3_8p_example",plan_digest="b"*64)
        self.assertEqual(self.waiter.find_progress([],**args), "QUEUED")
        marker = "LCI-R 123:2:dsv4_flash_a3_8p_example"
        comment = {"user":{"login":"ci-maintainer"}, "body":marker}
        self.assertEqual(self.waiter.find_progress([comment],**args),"RUNNING")
        for change in ({"attempt":3},{"test_id":"other_case"}):
            with self.subTest(change=change):
                self.assertEqual(self.waiter.find_progress([comment],**{**args,**change}),
                                 "QUEUED")
        comment["user"]["login"]="forged-author"
        self.assertEqual(self.waiter.find_progress([comment],**args),"QUEUED")
        self.assertEqual(
            self.waiter.heartbeat_line(status="QUEUED", current_case=args["test_id"],
                                  when=self.waiter.dt.datetime(2026,10,11,13,7,10,
                                                          tzinfo=self.waiter.dt.timezone.utc)),
            "2026/10/11 21:07:10 dsv4_flash_a3_8p_example CI queued.")

    def test_partial_matrix_rerun_without_artifact_fails_fast(self):
        import io
        with patch.object(self.waiter.urllib.request, "urlopen",
                          return_value=io.BytesIO(
                              b'{"artifacts":[{"name":"lite-ci-request-1",'
                              b'"expired":false}]}')):
            with self.assertRaisesRegex(ValueError, "Partial Matrix reruns"):
                self.waiter.ensure_attempt_artifact("test-token",1234,2)
        with patch.object(self.waiter.urllib.request, "urlopen",
                          return_value=io.BytesIO(
                              b'{"artifacts":[{"name":"lite-ci-request-2",'
                              b'"expired":false}]}')):
            self.waiter.ensure_attempt_artifact("test-token",1234,2)

    def test_org_only_repository_and_comment_author_binding(self):
        org = "torchtitan-npu/torchtitan-npu"
        self.assertEqual(self.prepare.REPO, org)
        self.assertEqual(self.waiter.REPO, org)
        cases = [{"test_id":"dsv4_flash_a3_8p_example","nnodes":1,"ngpu":8}]
        plan = self.prepare.create_plan(cases=cases,run_id=123,attempt=1,
                                   sha="a"*40,repo=org,pipeline="a3-8p")
        self.assertEqual(plan["repo"],org)
        report = {"schema":2,"repo":org,"run_id":123,"attempt":1,
                  "sha":"a"*40,"pipeline":"a3-8p","test_id":cases[0]["test_id"],
                  "plan_digest":"b"*64,"status":"PASS","exit_code":0,
                  "last_20_lines":[]}
        comment = {"user":{"login":"ci-maintainer"},
                   "body":f"Lite-CI-RESULT 123:1:{cases[0]['test_id']}\n"+
                          json.dumps(report)}
        args = dict(sha="a"*40,run_id=123,attempt=1,pipeline="a3-8p",
                    test_id=cases[0]["test_id"],plan_digest="b"*64)
        self.assertEqual(self.waiter.find_case_result([comment],**args),report)
        comment["user"]["login"]="forged-user"
        self.assertIsNone(self.waiter.find_case_result([comment],**args))
        with self.assertRaisesRegex(ValueError,"untrusted repository"):
            self.prepare.create_plan(cases=cases,run_id=123,attempt=1,
                                sha="a"*40,repo="other/repo",pipeline="a3-8p")
    def test_workflow_dispatch_guards_and_matrix_propagation(self):
        import yaml

        workflow_dir = ROOT / ".github/workflows"

        def read(name):
            return yaml.load((workflow_dir / name).read_text(), Loader=yaml.BaseLoader)

        def command_step(job, command):
            steps = [step for step in job["steps"]
                     if command in step.get("run", "")]
            self.assertEqual(len(steps), 1, f"Expected one {command} step")
            return steps[0]

        common = read("lite_actions_common.yml")
        self.assertIn("workflow_call", common["on"])
        self.assertIn("pipeline", common["on"]["workflow_call"]["inputs"])
        self.assertTrue({"prepare", "case", "smoke"}.issubset(common["jobs"]))
        self.assertEqual(common["permissions"], {"contents": "read", "actions": "read"})

        prepare_job = common["jobs"]["prepare"]
        case_job = common["jobs"]["case"]
        smoke_job = common["jobs"]["smoke"]
        for job in (prepare_job, smoke_job):
            self.assertIn("github.repository == 'torchtitan-npu/torchtitan-npu'", job["if"])
            self.assertIn("github.base_ref == 'master'", job["if"])
        self.assertEqual(case_job["needs"], "prepare")
        self.assertIn("needs.prepare.result == 'success'", case_job["if"])
        self.assertIn("fromJSON(needs.prepare.outputs.matrix)",
                      case_job["strategy"]["matrix"])
        self.assertEqual(case_job["strategy"]["fail-fast"], "false")

        producer = command_step(prepare_job, "prepare_request.py")
        self.assertIn("inputs.pipeline", producer["env"]["CI_PIPELINE"])
        self.assertIn("github.event.pull_request.head.sha", producer["env"]["CI_SOURCE_SHA"])
        self.assertIn("inputs.default_cases", producer["env"]["CI_TEST_CASES"])
        uploads = [step for step in prepare_job["steps"]
                   if step.get("uses", "").startswith("actions/upload-artifact@")]
        self.assertEqual(len(uploads), 1)
        self.assertEqual(uploads[0]["with"]["if-no-files-found"], "error")
        self.assertEqual(uploads[0]["with"]["path"], ".ci-request/ci-request.json")

        consumer = command_step(case_job, "wait_result.py")
        self.assertIn("--test-id", consumer["run"])
        self.assertIn("CI_PLAN_DIGEST", consumer["env"])
        self.assertIn("CI_SOURCE_SHA", consumer["env"])
        self.assertIn("CI_RESULT_AUTHOR", consumer["env"])
        self.assertIn("dispatcher-smoke", command_step(smoke_job, "wait_result.py")["run"])

        callers = (
            ("a3_8p_lite_actions.yml", "a3-8p", "a3_8p_tests", True),
            ("a3_16p_lite_actions.yml", "a3-16p", "a3_16p_tests", True),
            ("a5_64p_lite_actions.yml", "a5-64p", "a5_64p_tests", False),
            ("lite_actions_dispatcher_smoke.yml", "dispatcher-smoke", None, True),
        )
        for filename, pipeline, suite, enabled in callers:
            with self.subTest(pipeline=pipeline):
                workflow = read(filename)
                call_jobs = [job for job in workflow["jobs"].values()
                             if job.get("uses") == "./.github/workflows/lite_actions_common.yml"]
                self.assertEqual(len(call_jobs), 1)
                caller = call_jobs[0]
                self.assertEqual(caller["with"]["pipeline"], pipeline)
                self.assertEqual(workflow["permissions"], common["permissions"])
                self.assertIn("workflow_dispatch", workflow["on"])
                if suite:
                    self.assertIn(suite, caller["with"]["default_cases"])
                    self.assertIn("inputs.test_cases", caller["with"]["test_cases"])
                if not enabled:
                    self.assertNotIn("pull_request", workflow["on"])
                    self.assertIn("false", caller["if"])
                else:
                    self.assertEqual(workflow["on"]["pull_request"]["branches"], ["master"])

    def test_model_owned_plan_is_unchanged_after_reuse(self):
        selected=self.prepare.expand_cases([{"suite":"a3_16p_tests"}])
        self.assertEqual([c["test_id"] for c in selected],
                         ["dsv4_flash_a3_16p_example"])
        self.assertEqual((selected[0]["nnodes"],selected[0]["ngpu"]),(2,8))

    def test_lite_ci_sources_have_required_license_headers(self):
        # Guard the active Lite Actions source tree without enumerating an
        # arbitrary fixed number of files or relying on an unchanging layout.
        directories = (
            ROOT / ".github/scripts/lite_actions",
            ROOT / "tests/integration_tests/nightly_all_models_test",
            ROOT / "tests/integration_tests/tools/lite_actions",
            ROOT / "tests/unit_tests/tooling/lite_actions",
        )
        sources = [file for directory in directories
                   for file in directory.glob("*.py")]
        sources.extend((ROOT / ".github/workflows").glob("*lite_actions*.yml"))
        self.assertTrue(sources, "Lite Actions sources must be discoverable")
        for file in sources:
            with self.subTest(path=file.relative_to(ROOT)):
                lines = file.read_text().splitlines()
                if lines[0].startswith("#!"):
                    lines = lines[1:]
                self.assertRegex(lines[0], r"^# Copyright \(c\) 20\d{2} Huawei Technologies Co\., Ltd\.")
                self.assertIn("BSD-style license", "\n".join(lines[:6]))


if __name__ == "__main__":
    unittest.main()
