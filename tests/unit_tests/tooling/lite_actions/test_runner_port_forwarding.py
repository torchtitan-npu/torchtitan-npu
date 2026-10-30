# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU runner contracts for trusted port leases; no NPU or child launches."""

import pytest

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.nightly_all_models_test import runner


@pytest.fixture
def port_lease(monkeypatch):
    env = {
        'LITE_CI_MASTER_PORT': '16002',
        'LITE_CI_HCCL_IF_BASE_PORT': '20256',
        'LITE_CI_HCCL_HOST_PORT_RANGE': '24256-24319',
        'LITE_CI_HCCL_NPU_PORT_RANGE': '28256-28319',
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return {
        'MASTER_PORT': '16002',
        'HCCL_IF_BASE_PORT': '20256',
        'HCCL_HOST_SOCKET_PORT_RANGE': '24256-24319',
        'HCCL_NPU_SOCKET_PORT_RANGE': '28256-28319',
    }


def _case(nnodes):
    return OverrideDefinitions(
        test_name='ci_sample', nnodes=nnodes, ngpu=8,
        train_script='unused.sh', override_args=[('--training.steps', '5')],
        env_vars={
            'MASTER_PORT': '6316',
            'HCCL_IF_BASE_PORT': '30160',
            'HCCL_HOST_SOCKET_PORT_RANGE': '30000-30063',
            'HCCL_NPU_SOCKET_PORT_RANGE': '50000-50063',
            'KEEP_RECIPE_ENV': 'present',
        },
    )


def test_single_node_runner_receives_trusted_ports_over_recipe_defaults(
    monkeypatch, tmp_path, port_lease,
):
    from tests.integration_tests import run_tests as test_executor

    seen = []
    monkeypatch.setattr(test_executor, 'run_tests', lambda *args, **kwargs: seen.append((args, kwargs)))
    original = _case(nnodes=1)

    runner.run_single(original, output_dir=tmp_path / 'single')

    assert len(seen) == 1
    supplied = seen[0][0][1][0]
    assert {key: supplied.env_vars[key] for key in port_lease} == port_lease
    assert supplied.env_vars['KEEP_RECIPE_ENV'] == 'present'
    assert original.env_vars['MASTER_PORT'] == '6316'  # no mutation of shared recipe


def test_multi_node_runner_passes_trusted_ports_to_actual_subprocess(
    monkeypatch, tmp_path, port_lease,
):
    seen = []

    def capture(command, *, env):
        seen.append((command, env))
        return 0

    monkeypatch.setattr(runner.subprocess, 'call', capture)
    monkeypatch.setenv('NODE_IPS', '192.0.2.1,192.0.2.2')
    monkeypatch.setenv('NGPU', '8')
    original = _case(nnodes=2)

    with pytest.raises(SystemExit) as result:
        runner.run_distributed(original, phase='launch', output_dir=tmp_path / 'multi')

    assert result.value.code == 0
    assert len(seen) == 1
    command, environment = seen[0]
    assert command[0:2] == ['bash', original.train_script]
    assert {key: environment[key] for key in port_lease} == port_lease
    assert environment['KEEP_RECIPE_ENV'] == 'present'
    assert original.env_vars['MASTER_PORT'] == '6316'
