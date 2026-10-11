"""CPU-only GitHub Actions Inputs -> bound JSON Artifact contract."""
from __future__ import annotations
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[4]/'.github/scripts/lite_actions/prepare_request.py'
spec=importlib.util.spec_from_file_location('prepare_ci_request',SCRIPT)
prepare=importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
CASE={'test_id':'dsv4_flash_a3_8p_example'}

class InputsArtifactTests(unittest.TestCase):
    def test_valid_multiple_cases(self):
        self.assertEqual(prepare.parse_cases(json.dumps([CASE,CASE])),[CASE,CASE])

    def test_reject_unsafe_ids_and_removed_parameter_fields(self):
        for test_id in ('../bin/sh','test;curl', 'Invalid-ID'):
            with self.subTest(test_id=test_id),self.assertRaises(ValueError):
                prepare.parse_cases(json.dumps([{**CASE,'test_id':test_id}]))
        with self.assertRaises(ValueError):
            prepare.parse_cases(json.dumps([{**CASE,'params':{}}]))

    def test_workflow_budget_blocks_too_many_cases(self):
        with patch.dict(os.environ, {'CI_MAX_CASES':'2'}):
            with self.assertRaises(ValueError):
                prepare.parse_cases(json.dumps([CASE]*3))
        with patch.dict(os.environ, {'CI_MAX_CASES':'1'}):
            with self.assertRaises(ValueError):
                prepare.parse_cases(json.dumps([CASE,CASE]))

    def test_artifact_identity_is_git_run_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            env={'CI_TEST_CASES':json.dumps([CASE]),'GITHUB_RUN_ID':'101',
                 'GITHUB_RUN_ATTEMPT':'2','GITHUB_SHA':'a'*40}
            with patch.dict(os.environ,env):
                old=Path.cwd()
                try:
                    os.chdir(td)
                    prepare.main()
                    value=json.loads(Path('.ci-request/ci-request.json').read_text())
                    self.assertEqual(value['cases'],[CASE])
                    self.assertEqual(value['run_id'],101)
                    self.assertEqual(value['attempt'],2)
                finally:
                    os.chdir(old)
            self.assertEqual(Path.cwd(),old)
            self.assertTrue(Path('tests/unit_tests/tooling/lite_actions/test_request.py').exists())

if __name__=='__main__':unittest.main()
