"""Generic Lite Actions adapter contracts, independent of the current model recipes."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration_tests import OverrideDefinitions
from tests.integration_tests.tools.lite_actions import entrypoint


def _case(name: str, *, nnodes: int = 1, ngpu: int = 8,
          disabled: bool = False) -> OverrideDefinitions:
    return OverrideDefinitions(test_name=name, nnodes=nnodes, ngpu=ngpu,
                               disabled=disabled, env_vars={"TEST_MODE": name})


def _fake_modules(monkeypatch, suites):
    """Exercise real discovery while replacing only the import/discovery boundary."""
    package = SimpleNamespace(__path__=["/synthetic"])
    modules = {entrypoint.PACKAGE: package}
    for name, cases in suites.items():
        modules[entrypoint.PACKAGE + "." + name] = SimpleNamespace(
            build_test_list=lambda values=cases: values)
    monkeypatch.setattr(entrypoint.importlib, 'import_module', lambda name: modules[name])
    monkeypatch.setattr(entrypoint.pkgutil, 'iter_modules', lambda _: [
        SimpleNamespace(name=name) for name in reversed(tuple(suites))])


def test_catalog_select_and_inspect_are_generic(monkeypatch, capsys):
    alpha, beta, gamma = _case('case_alpha'), _case('case_beta'), _case('case_gamma', nnodes=2)
    _fake_modules(monkeypatch, {'synthetic_tests':[alpha, beta],
                                'distributed_tests':[gamma]})
    groups = entrypoint.catalog()
    assert list(groups) == ['distributed_tests', 'synthetic_tests']
    assert entrypoint.select(groups, suite='synthetic_tests') == [alpha, beta]
    assert entrypoint.select(groups, test_id='case_beta') == [beta]
    with pytest.raises(ValueError, match='unknown'):
        entrypoint.select(groups, test_id='case_absent')
    with pytest.raises(ValueError, match='unknown'):
        entrypoint.select(groups, suite='not_registered')
    with pytest.raises(ValueError, match='exactly one'):
        entrypoint.select(groups, test_id='case_alpha', suite='synthetic_tests')
    monkeypatch.setattr(sys, 'argv', ['entrypoint', 'inspect', '--suite', 'synthetic_tests'])
    entrypoint.main()
    data = json.loads(capsys.readouterr().out)
    assert data == [
        {'test_id':'case_alpha','ngpu':8,'nnodes':1,'env_vars':{'TEST_MODE':'case_alpha'}},
        {'test_id':'case_beta','ngpu':8,'nnodes':1,'env_vars':{'TEST_MODE':'case_beta'}},
    ]


@pytest.mark.parametrize('name,cases,reason', [
    ('synthetic_tests', [], 'empty'),
    ('synthetic_tests', [_case('case_alpha'), _case('case_alpha')], 'duplicate'),
    ('synthetic_tests', [_case('case_alpha', disabled=True)], 'disabled'),
    ('synthetic_tests', [_case('bad-name')], 'invalid'),
    ('synthetic_tests', [_case('case_alpha', ngpu=0)], 'topology'),
    ('synthetic_tests', [_case('case_alpha', nnodes=0)], 'topology'),
])
def test_catalog_rejects_invalid_definitions(monkeypatch, name, cases, reason):
    _fake_modules(monkeypatch, {name:cases})
    with pytest.raises(ValueError, match=reason):
        entrypoint.catalog()


def test_catalog_rejects_duplicates_across_modules(monkeypatch):
    _fake_modules(monkeypatch, {
        'synthetic_tests':[_case('case_alpha')],
        'other_tests':[_case('case_alpha')]})
    with pytest.raises(ValueError, match='duplicate'):
        entrypoint.catalog()


@pytest.mark.parametrize('nnodes,phase,runner', [
    (1, 'launch', 'run_single'),
    (2, 'launch', 'run_distributed'),
    (2, 'verify', 'run_distributed'),
])
def test_launch_and_verify_forward_original_case(monkeypatch, tmp_path, nnodes, phase, runner):
    from tests.integration_tests.nightly_all_models_test import runner as impl
    case = _case('case_alpha', nnodes=nnodes)
    _fake_modules(monkeypatch, {'synthetic_tests':[case]})
    single, multi = [], []
    monkeypatch.setattr(impl, 'run_single', lambda *a, **kw: single.append((a, kw)))
    monkeypatch.setattr(impl, 'run_distributed', lambda *a, **kw: multi.append((a, kw)))
    output = tmp_path / 'output'
    monkeypatch.setattr(sys, 'argv', ['entrypoint', phase, '--test-id', 'case_alpha',
                                       '--output-dir', str(output)])
    entrypoint.main()
    expected = ((case,), {'output_dir': output}) if runner == 'run_single' else (
        (case,), {'phase': phase, 'output_dir': output})
    assert (single if runner == 'run_single' else multi) == [expected]
    assert (multi if runner == 'run_single' else single) == []
    assert entrypoint.os.environ['TEST_MODE'] == 'case_alpha'
    monkeypatch.delenv('TEST_MODE', raising=False)


def test_single_verify_has_no_training_side_effect(monkeypatch, tmp_path):
    from tests.integration_tests.nightly_all_models_test import runner as impl
    _fake_modules(monkeypatch, {'synthetic_tests':[_case('case_alpha')]})
    monkeypatch.setattr(impl, 'run_single', lambda *_a, **_kw: pytest.fail('training called'))
    monkeypatch.setattr(sys, 'argv', ['entrypoint', 'verify', '--test-id', 'case_alpha',
                                      '--output-dir', str(tmp_path)])
    entrypoint.main()
    monkeypatch.delenv('TEST_MODE', raising=False)
