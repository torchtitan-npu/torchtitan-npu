# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schema-3 behavior regressions shared by producer and consumer."""

import copy
import unittest
from types import SimpleNamespace
from tests.integration_tests.tools.lite_actions.contract import normalize, create_plan, parse_plan, digest

D = dict(worker="blue", target_pool="a3_shared_blue", docker="image:tag")
RUN = dict(id=19, run_attempt=2, head_sha="a" * 40)


def definition(**kw):
    return SimpleNamespace(
        **dict(test_name="dsv4_flash_a3_8p_example", test_descr="DSV4 real 8P", nnodes=1, ngpu=8, timeout=7200, **kw)
    )


def plan_for(items=None):
    return create_plan(
        items or [normalize(definition(), D, 0)],
        run_id=19,
        attempt=2,
        sha="a" * 40,
        workflow="nightly_fulltests.yml",
        defaults=D,
    )


class ContractV3Tests(unittest.TestCase):
    def test_workflow_defaults(self):
        c = normalize(definition(), D, 0)
        self.assertEqual(
            (c["worker"], c["target_pool"], c["resources"]),
            ("blue", "a3_shared_blue", dict(nodes=1, devices_per_node=8)),
        )

    def test_fieldwise_python_override(self):
        c = normalize(definition(ci_overrides={"docker": "new:tag"}), D, 0)
        self.assertEqual(c["docker"], "new:tag")
        self.assertEqual(c["worker"], "blue")

    def test_reject_empty_unknown_and_control_override(self):
        for bad in [{"docker": ""}, {"command": "rm x"}, {"worker": "green"}, {"docker": "a\nb"}]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize(definition(ci_overrides=bad), D, 0)

    def test_probe_has_no_training_defaults(self):
        c = normalize(definition(ci_kind="worker_probe", ci_overrides={"worker": "yellow"}), D, 0)
        self.assertIsNone(c["target_pool"])
        self.assertIsNone(c["docker"])
        self.assertEqual(c["resources"]["nodes"], 0)

    def test_probe_rejects_explicit_resource_fields(self):
        with self.assertRaises(ValueError):
            normalize(definition(ci_kind="worker_probe", ci_overrides={"target_pool": "a3_shared_blue"}), D, 0)

    def test_identity_and_digest(self):
        p = plan_for()
        self.assertEqual(parse_plan(p, RUN, p["workflow"]), p["cases"])
        for key, val in [("sha", "b" * 40), ("attempt", 3), ("repo", "another/repo")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_plan({**p, key: val}, RUN, p["workflow"])

    def test_every_execution_field_is_digest_bound(self):
        for key, val in [
            ("worker", "yellow"),
            ("docker", "other:tag"),
            ("target_pool", "a3_shared_yellow"),
            ("resources", dict(nodes=2, devices_per_node=8)),
        ]:
            p = copy.deepcopy(plan_for())
            p["cases"][0][key] = val
            with self.subTest(key=key), self.assertRaises(ValueError):
                parse_plan(p, RUN, p["workflow"])

    def test_duplicate_and_noncanonical_case_rejected(self):
        p = plan_for()
        p["cases"].append(copy.deepcopy(p["cases"][0]))
        p["plan_digest"] = digest({k: v for k, v in p.items() if k != "plan_digest"})
        with self.assertRaises(ValueError):
            parse_plan(p, RUN, p["workflow"])

    def test_boolean_resource_count_rejected(self):
        x = definition()
        x.ngpu = True
        with self.assertRaises(ValueError):
            normalize(x, D, 0)

    def test_unknown_kind_rejected(self):
        with self.assertRaises(ValueError):
            normalize(definition(ci_kind="shell"), D, 0)
