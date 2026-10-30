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
            "dispatch_smoke_a5_8p": 8,
            "dispatch_smoke_32p": 32,
        })
        self.assertTrue(is_dispatch_smoke(eight))
        self.assertTrue(is_dispatch_smoke(sixteen))

    def test_a5_smokes_are_registered_with_exact_topology_and_overrides(self):
        cases = catalog()
        for name, nodes, timeout in (("dispatch_smoke_a5_8p", 1, 120), ("dispatch_smoke_32p", 4, 150)):
            case = select(cases, test_id=name)[0]
            self.assertEqual((case.nnodes, case.ngpu), (nodes, 8))
            self.assertEqual(case.timeout, timeout)
            self.assertEqual(case.ci_kind, "npu_smoke")
            self.assertEqual(case.ci_overrides, {"worker": "yellow", "target_pool": "a5_shared_yellow", "docker": "vllm-kimi-wangjinyin"})
            self.assertFalse(case.check_loss)
            self.assertTrue(is_dispatch_smoke(case))

    def _capture_smoke_launch_env(self, *, test_id, network_env):
        import os
        import socket
        import tempfile
        from pathlib import Path
        from unittest.mock import patch
        from tests.integration_tests.nightly_all_models_test.runner import run_dispatch_smoke

        case = select(catalog(), test_id=test_id)[0]
        node_ips = ["127.0.0.1"] * case.nnodes
        if case.nnodes > 1:
            node_ips = [f"127.0.0.{i + 1}" for i in range(case.nnodes)]
        env = {
            "NNODES": str(case.nnodes),
            "NGPU": str(case.ngpu),
            "NODE_IPS": ",".join(node_ips),
            "LITE_CI_MASTER_PORT": "31001",
            **network_env,
        }
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.dict(os.environ, env, clear=True),
            patch.object(socket, "getaddrinfo", return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", (node_ips[0], 0)),
            ]),
            patch("subprocess.check_output", return_value=node_ips[0]),
            patch("subprocess.run") as launch,
        ):
            run_dispatch_smoke(case, output_dir=Path(folder) / "smoke")
            launch.assert_called_once()
            self.assertEqual(launch.call_args.kwargs["timeout"], case.timeout)
            return launch.call_args.kwargs["env"]

    def test_multinode_smoke_uses_trusted_interface_fallbacks(self):
        env = self._capture_smoke_launch_env(
            test_id="dispatch_smoke_32p",
            network_env={"Network_Interface": "enp34s0f1"},
        )
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "enp34s0f1")
        self.assertEqual(env["HCCL_SOCKET_IFNAME"], "enp34s0f1")
        self.assertEqual(env["HCCL_IF_BASE_PORT"], "30000")
        self.assertEqual(env["LITE_SMOKE_INIT_TIMEOUT"], "90")

    def test_a5_32p_hccl_if_ip_matches_local_node_ip(self):
        env = self._capture_smoke_launch_env(
            test_id="dispatch_smoke_32p",
            network_env={"Network_Interface": "enp34s0f1"},
        )
        self.assertEqual(env["HCCL_IF_IP"], "127.0.0.1")

    def test_a5_32p_hccl_if_ip_does_not_override_explicit_value(self):
        env = self._capture_smoke_launch_env(
            test_id="dispatch_smoke_32p",
            network_env={
                "Network_Interface": "enp34s0f1",
                "HCCL_IF_IP": "192.0.2.10",
            },
        )
        self.assertEqual(env["HCCL_IF_IP"], "192.0.2.10")

    def test_a3_16p_hccl_if_ip_semantics_unchanged(self):
        env = self._capture_smoke_launch_env(
            test_id="dispatch_smoke_16p",
            network_env={"Network_Interface": "enp34s0f1"},
        )
        self.assertNotIn("HCCL_IF_IP", env)

    def test_multinode_smoke_preserves_explicit_network_settings(self):
        env = self._capture_smoke_launch_env(
            test_id="dispatch_smoke_32p",
            network_env={
                "Network_Interface": "enp34s0f1",
                "GLOO_SOCKET_IFNAME": "bond0",
                "HCCL_SOCKET_IFNAME": "eth2",
                "HCCL_IF_BASE_PORT": "31000",
            },
        )
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "bond0")
        self.assertEqual(env["HCCL_SOCKET_IFNAME"], "eth2")
        self.assertEqual(env["HCCL_IF_BASE_PORT"], "31000")
        self.assertEqual(env["LITE_SMOKE_INIT_TIMEOUT"], "90")

    def test_single_node_smoke_keeps_default_init_timeout(self):
        env = self._capture_smoke_launch_env(
            test_id="dispatch_smoke_a5_8p",
            network_env={"Network_Interface": "enp34s0f1"},
        )
        self.assertNotIn("LITE_SMOKE_INIT_TIMEOUT", env)
        self.assertEqual(env["GLOO_SOCKET_IFNAME"], "enp34s0f1")
        self.assertEqual(env["HCCL_SOCKET_IFNAME"], "enp34s0f1")

    def test_smoke_cases_do_not_enable_loss_or_training_steps(self):
        cases = catalog()
        for test_id in ("dispatch_smoke_8p", "dispatch_smoke_16p"):
            case = select(cases, test_id=test_id)[0]
            self.assertFalse(case.check_loss)
            self.assertIsNone(case.expected_steps)
            self.assertEqual(case.timeout, 75)


if __name__ == "__main__":
    unittest.main()
