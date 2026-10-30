# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only shell CLI contracts for DeepSeek-V4 Flash model-specific nightly cases."""
import os
from pathlib import Path
import subprocess
from dataclasses import replace

import pytest

from tests.integration_tests.nightly_all_models_test import a3_8p_tests, a3_16p_tests, a5_64p_tests

ROOT = Path(__file__).resolve().parents[4]


def expanded_argv(test, tmp_path):
    stub = tmp_path / 'scripts'
    stub.mkdir()
    name = 'run_train_multinodes.sh' if test.nnodes > 1 else 'run_train.sh'
    (stub / name).write_text('printf "%s\\n" "$@" > "$CASE_ARGS_FILE"\n')
    env = {**os.environ, **(test.env_vars or {}),
           'CASE_ARGS_FILE': str(tmp_path / 'args.txt'),
           'NODE_IPS': ','.join(f'192.0.2.{i}' for i in range(1, test.nnodes+1)),
           'NGPU': str(test.ngpu)}
    subprocess.run(['bash', str(ROOT / test.train_script), *test.override_args[0]],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    return (tmp_path / 'args.txt').read_text().splitlines()


def last(argv, flag):
    return argv[max(i for i, arg in enumerate(argv) if arg == flag) + 1]


def imports(argv):
    i = max(i for i, arg in enumerate(argv) if arg == '--override.imports')
    result = []
    for item in argv[i+1:]:
        if item.startswith('--'):
            break
        result.append(item)
    return result


@pytest.mark.parametrize(
    'case_id,optimizer,enabled_override,disabled_override',
    [
        ('dsv4_flash_a3_8p_example', 'Muon',
         'torchtitan_npu.override.common.optimizer.swap_optimizer',
         'torchtitan_npu.override.common.optimizer.virtual'),
        ('dsv4_flash_a3_8p_adamw', 'AdamW',
         'torchtitan_npu.override.common.optimizer.virtual',
         'torchtitan_npu.override.common.optimizer.swap_optimizer'),
    ],
    ids=['muon', 'adamw'],
)
def test_a3_8p_optimizer_recipe_uses_selected_override(
    tmp_path, case_id, optimizer, enabled_override, disabled_override,
):
    matches = [case for case in a3_8p_tests.build_test_list()
               if case.test_name == case_id]
    assert len(matches) == 1, f'{case_id} must be registered exactly once'
    case = matches[0]

    argv = expanded_argv(case, tmp_path)
    selected_imports = imports(argv)

    assert last(argv, '--training.steps') == '5'
    assert last(argv, '--optimizer.name') == optimizer
    assert last(argv, '--training.seq-len') == '4096'
    assert '--compile.no-enable' in argv
    assert 'torchtitan_npu.override.common.rms_norm.asc' in selected_imports
    assert selected_imports.count(enabled_override) == 1
    assert disabled_override not in selected_imports


def test_a3_16p_effective_recipe(tmp_path):
    case = a3_16p_tests.build_test_list()[0]
    assert case.env_vars["HF_HOME"].startswith("/mnt/share/ci_tests/")
    assert case.env_vars["HF_DATASETS_CACHE"].startswith(case.env_vars["HF_HOME"] + "/")
    argv = expanded_argv(case, tmp_path)
    assert last(argv, '--parallelism.expert-parallel-degree') == '16'
    assert case.env_vars['TORCHINDUCTOR_NPU_BACKEND'] == 'ascendc'
    assert last(argv, '--optimizer.name') == 'AdamW'
    assert '--checkpoint.no-enable' in argv
    assert 'torchtitan_npu.override.common.optimizer.swap_optimizer' not in imports(argv)
    assert 'torchtitan_npu.override.common.rms_norm.asc' in imports(argv)


def test_a5_model_recipe_passes_initial_checkpoint_options_to_launcher(tmp_path):
    # CPU contract only: the A5 hardware pool remains disabled.
    case = next(test for test in a5_64p_tests.build_test_list()
                if test.test_name == 'dsv4_pro_a5_64p')
    # The disabled A5 pool has an unconfigured checkpoint path. Provide a
    # sentinel without mutating the registered recipe, then capture the real
    # shell-expanded argument forwarded to the training launcher.
    checkpoint = str(tmp_path / 'expected_initial_checkpoint')
    configured = replace(case, env_vars={**case.env_vars,
                                         'CKPT_INIT_LOAD_PATH': checkpoint})

    argv = expanded_argv(configured, tmp_path)

    assert '--checkpoint.enable' in argv
    assert '--checkpoint.load-only' in argv
    assert '--checkpoint.initial-load-in-hf' in argv
    assert last(argv, '--checkpoint.initial-load-path') == checkpoint
    assert case.env_vars['CKPT_INIT_LOAD_PATH'] == ''  # A5 still not deployable
