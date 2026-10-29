# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Licensed under the BSD-style license found in the LICENSE file.

import unittest

from tests.integration_tests.tools.lite_actions.entrypoint import catalog, select
from tests.integration_tests.nightly_all_models_test.runner import (
    SMOKE_WORLD,
    is_dispatch_smoke,
)


class LiteActionsDispatchSmokeTests(unittest.TestCase):
    def test_8p_and_16p_smokes_are_registered_with_exact_topology(self):
        cases = catalog()
        eight = select(cases, test_id="dispatch_smoke_8p")[0]
        sixteen = select(cases, test_id="dispatch_smoke_16p")[0]
        self.assertEqual((eight.nnodes, eight.ngpu), (1, 8))
        self.assertEqual((sixteen.nnodes, sixteen.ngpu), (2, 8))
        self.assertEqual(SMOKE_WORLD, {
            "dispatch_smoke_8p": 8,
            "dispatch_smoke_16p": 16,
        })
        self.assertTrue(is_dispatch_smoke(eight))
        self.assertTrue(is_dispatch_smoke(sixteen))

    def test_smoke_cases_do_not_enable_loss_or_training_steps(self):
        cases = catalog()
        for test_id in ("dispatch_smoke_8p", "dispatch_smoke_16p"):
            case = select(cases, test_id=test_id)[0]
            self.assertFalse(case.check_loss)
            self.assertIsNone(case.expected_steps)
            self.assertEqual(case.timeout, 75)


if __name__ == "__main__":
    unittest.main()
