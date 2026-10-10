"""CPU-only shell CLI contracts for DeepSeek-V4 Flash model-specific nightly cases."""
import os
from pathlib import Path
import subprocess

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


def test_a3_muon_and_adamw_effective_recipe(tmp_path):
    muon, adamw = a3_8p_tests.build_test_list()
    swap = 'torchtitan_npu.override.common.optimizer.swap_optimizer'
    virtual = 'torchtitan_npu.override.common.optimizer.virtual'
    npu = 'torchtitan_npu.override.common.rms_norm.asc'
    for i, case in enumerate((muon, adamw)):
        folder = tmp_path / str(i)
        folder.mkdir()
        argv = expanded_argv(case, folder)
        assert last(argv, '--training.steps') == '5'
        assert '--compile.no-enable' in argv
        assert npu in imports(argv)
        assert last(argv, '--optimizer.name') == ('Muon' if i == 0 else 'AdamW')
        assert last(argv, '--training.seq-len') == '4096'
        effective = imports(argv)
        assert (swap in effective) == (i == 0)
        assert (virtual in effective) == (i == 1)
        assert effective.count(virtual) == (1 if i == 1 else 0)
        # Both replace the same config node and must never co-exist.
        assert not ({virtual, swap} <= set(effective))


def test_a3_16p_effective_recipe(tmp_path):
    case = a3_16p_tests.build_test_list()[0]
    argv = expanded_argv(case, tmp_path)
    assert last(argv, '--parallelism.expert-parallel-degree') == '16'
    assert case.env_vars['TORCHINDUCTOR_NPU_BACKEND'] == 'ascendc'
    assert last(argv, '--optimizer.name') == 'AdamW'
    assert '--checkpoint.no-enable' in argv
    assert 'torchtitan_npu.override.common.optimizer.swap_optimizer' not in imports(argv)
    assert 'torchtitan_npu.override.common.rms_norm.asc' in imports(argv)


def test_a5_model_recipe_keeps_explicit_initial_checkpoint(tmp_path):
    # The A5 case is still disabled at the dispatcher. Removing the old
    # Deleting the redundant checkpoint preflight flag cannot disable loading.
    case = a5_64p_tests.build_test_list()[0]
    assert case.env_vars['CKPT_INIT_LOAD_PATH'] == ''
    script = (ROOT / case.train_script).read_text()
    assert '--checkpoint.enable' in script
    assert '--checkpoint.load-only' in script
    assert '--checkpoint.initial-load-path ${CKPT_INIT_LOAD_PATH}' in script
    assert '--checkpoint.initial-load-in-hf' in script
