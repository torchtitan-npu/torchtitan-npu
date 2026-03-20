# Test Design

## Purpose
This document explains the current test layering in `torchtitan-npu` and helps contributors decide where a new test should live.

The goals are simple:
- Keep unit tests fast and hardware-independent
- Keep smoke tests focused on real execution paths
- Make test placement easy to understand for new contributors

## Test Layers
| Layer | Directory | NPU Required | Use Cases |
|---|---|---|---|
| Function UT | `tests/unit_tests/functions/` | No | Pure functions, config parsing, helpers, validation logic |
| Module UT | `tests/unit_tests/modules/` | No | Wrappers, checkpoint logic, distributed initialization logic |
| Converter UT | `tests/unit_tests/converters/` | No | Converter registration, replacement, and mapping logic |
| Patch UT | `tests/unit_tests/patches/` | No | Patch activation, wiring, and small patch behavior |
| Feature Smoke | `tests/smoke_tests/features/` | Yes | Real NPU feature paths, fused ops, wrapper execution chains |
| Model Parallel Smoke | `tests/smoke_tests/model_parallel/` | Yes | CP/TP/EP behavior, mesh setup, DTensor, model-parallel scenarios |

## How to Choose a Layer
- If the test can run without NPU, prefer UT
- If the value of the test depends on real NPU execution, use smoke
- If the change is about mesh, shard, placement, or DTensor behavior, use model-parallel smoke
- If the change is about a small helper or pure transformation, keep it in UT

## What "Smoke" Means Here
In this repository, smoke tests are not import checks. They are integration-style checks over real execution paths.

`build.sh -s` runs two parts by default:
- Core smoke: minimal end-to-end training path validation
- Extended smoke: local feature and model-parallel smoke suites

Upstream smoke is kept as a separate targeted entry. It is heavier, takes longer, and may be affected by low-level hardware issues, so it is not part of the default smoke path.

## Rules for Adding Tests
1. Prefer real behavior validation over import-only checks.
2. Do not add placeholder tests that only make the suite look larger.
3. Keep UT fully hardware-independent.
4. Use smoke only when the behavior matters on real NPU.
5. If a test depends on external artifacts or a special runtime setup, state that clearly.

## Readability Guidelines
- Use test names that describe behavior and expected outcome directly
- Keep setup short and easy to scan
- Use blank lines to separate Arrange, Act, and Assert when it helps readability
- Let assertions show the exact behavior being protected

Avoid:
- Long per-test docstrings
- Comments that just restate the test name
- Decorative section comments with no real meaning

## Build Entry Points
- `build.sh -u`: run all unit tests
- `build.sh -s`: run all smoke tests

Useful variants:
- `ONLY_CORE_SMOKE=true build.sh -s`
- `ONLY_EXTENDED_SMOKE=true build.sh -s`
- `ONLY_UPSTREAM_SMOKE=true build.sh -s`

## Pre-Submission Checklist
1. Is the test in the right directory?
2. Does it validate real behavior instead of a placeholder path?
3. Can a new contributor understand the test quickly?
4. If the change affects test execution or test usage, did you update the docs?
