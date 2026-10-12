# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
CASE={'test_id':'dsv4_flash_a3_8p_example'}

class InputsArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        spec = importlib.util.spec_from_file_location('prepare_ci_request', SCRIPT)
        cls.prepare = importlib.util.module_from_spec(spec)
        with patch.dict(os.environ, {'GITHUB_REPOSITORY':'torchtitan-npu/torchtitan-npu'}):
            spec.loader.exec_module(cls.prepare)

    def test_valid_multiple_cases(self):
        self.assertEqual(self.prepare.parse_cases(json.dumps([CASE,CASE])),[CASE,CASE])

    def test_reject_unsafe_ids_and_removed_parameter_fields(self):
        for test_id in ('../bin/sh','test;curl', 'Invalid-ID'):
            with self.subTest(test_id=test_id),self.assertRaises(ValueError):
                self.prepare.parse_cases(json.dumps([{**CASE,'test_id':test_id}]))
        with self.assertRaises(ValueError):
            self.prepare.parse_cases(json.dumps([{**CASE,'params':{}}]))

    def test_selectors_are_not_arbitrarily_capped(self):
        many = [{'test_id': f'case_{i:03d}'} for i in range(40)]
        self.assertEqual(self.prepare.parse_cases(json.dumps(many)), many)
        with self.assertRaises(ValueError):
            self.prepare.parse_cases('[]')

    def test_artifact_identity_is_git_run_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            env = {'CI_TEST_CASES': json.dumps([CASE]), 'GITHUB_RUN_ID': '101',
                   'GITHUB_RUN_ATTEMPT': '2', 'GITHUB_SHA': 'a'*40,
                   'GITHUB_REPOSITORY': 'torchtitan-npu/torchtitan-npu',
                   'CI_PIPELINE': 'a3-8p',
                   'GITHUB_OUTPUT': str(Path(td) / 'gh-output.txt')}
            with patch.dict(os.environ, env):
                old = Path.cwd()
                try:
                    os.chdir(td)
                    self.prepare.main()
                    value=json.loads(Path('.ci-request/ci-request.json').read_text())
                    self.assertEqual([r['test_id'] for r in value['cases']],
                                     [CASE['test_id']])
                    self.assertEqual(value['schema'], 2)
                    self.assertEqual(value['run_id'], 101)
                    self.assertEqual(value['attempt'], 2)
                    self.assertEqual(value['repo'], env['GITHUB_REPOSITORY'])
                finally:
                    os.chdir(old)
            self.assertEqual(Path.cwd(), old)
            self.assertTrue(Path('tests/unit_tests/tooling/lite_actions/test_request.py').exists())


if __name__=='__main__':unittest.main()
