#!/usr/bin/env python3
"""Lite Actions adapter: trusted case definitions and fixed SHA selectors."""
from __future__ import annotations
import argparse
import importlib
import json
import os
from pathlib import Path
import pkgutil
import re

PACKAGE="tests.integration_tests.nightly_all_models_test"
SAFE=re.compile(r"[a-z][a-z0-9_]{0,79}\Z")

def catalog() -> dict[str,list]:
    pkg=importlib.import_module(PACKAGE)
    found={}
    names=set()
    for info in sorted(pkgutil.iter_modules(pkg.__path__),key=lambda x:x.name):
        if not info.name.endswith('_tests') or not SAFE.fullmatch(info.name):continue
        module=importlib.import_module(PACKAGE+'.'+info.name)
        tests=module.build_test_list()
        if not isinstance(tests,list) or not tests:raise ValueError('empty test suite')
        for test in tests:
            if not SAFE.fullmatch(test.test_name) or test.test_name in names or test.disabled:
                raise ValueError('duplicate/invalid/disabled CI case')
            if not isinstance(test.nnodes,int) or test.nnodes<1 or not isinstance(test.ngpu,int) or test.ngpu<1:
                raise ValueError('invalid topology')
            names.add(test.test_name)
        found[info.name]=tests
    return found

def select(catalogue:dict, *,test_id:str|None=None,suite:str|None=None):
    if (test_id is None)==(suite is None):raise ValueError('select exactly one ID or suite')
    if suite is not None:
        if suite not in catalogue:raise ValueError('unknown CI suite')
        return catalogue[suite]
    matches=[x for group in catalogue.values() for x in group if x.test_name==test_id]
    if len(matches)!=1:raise ValueError('unknown or duplicated CI test')
    return matches

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('inspect','launch','verify'))
    sel=p.add_mutually_exclusive_group(required=True)
    sel.add_argument('--test-id');sel.add_argument('--suite')
    p.add_argument('--output-dir',type=Path)
    a=p.parse_args()
    if a.phase!='inspect' and (a.test_id is None or a.output_dir is None):
        p.error('launch/verify require an explicit test ID and output directory')
    try: selected=select(catalog(),test_id=a.test_id,suite=a.suite)
    except ValueError as exc:p.error(str(exc))
    if a.phase=='inspect':
        print(json.dumps([{'test_id': t.test_name, 'ngpu': t.ngpu,
           'nnodes': t.nnodes, 'env_vars': dict(t.env_vars or {})} for t in selected],
           separators=(',', ':')))
        return
    test=selected[0]
    # Also support direct, manual invocation of the fixed model-owned entrypoint.
    # Remote SSH has already sourced the case's toolkit before Python starts.
    os.environ.update(test.env_vars or {})
    from tests.integration_tests.nightly_all_models_test.runner import run_single,run_distributed
    if test.nnodes==1:
        if a.phase=='launch':run_single(test,output_dir=a.output_dir)
    else:
        run_distributed(test, phase=a.phase, output_dir=a.output_dir)

if __name__=='__main__':main()
