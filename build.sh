# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh

pip install -r requirements.txt
pip install -r requirements_dev.txt

# Global variable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
OUTPUT_DIR="${PROJECT_ROOT}/output"
REPORT_DIR="${PROJECT_ROOT}/test_reports"
INTEGRATION_REPORT_DIR="${PROJECT_ROOT}/test_reports/integration_tests"
TITAN_VERSION="v0.2.2"
TITAN_DIR="${PROJECT_ROOT}/third_party/torchtitan"
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}


# Run torchtitan upstream unit tests (with NPU patches applied)
run_upstream_ut() {
    echo "Running torchtitan upstream unit tests..."

    mkdir -p "$REPORT_DIR"

    # Ensure torchtitan_npu is installed (applies patches on import)
    if ! python3 -c "import torchtitan_npu" 2>/dev/null; then
        python3 -m pip install -e .
    fi

    # Clone torchtitan source if not exists
    if [ ! -d "$TITAN_DIR" ]; then
        echo "Cloning torchtitan source..."
        mkdir -p third_party
        git clone --branch $TITAN_VERSION --depth 1 \
            https://gitcode.com/GitHub_Trending/to/torchtitan.git $TITAN_DIR
    fi

    # Create conftest.py in torchtitan test dir to ensure import torchtitan_npu before each test
    local titan_test_dir="${TITAN_DIR}/tests/unit_tests"
    local conftest_file="${titan_test_dir}/conftest.py"

    if [[ ! -d "$titan_test_dir" ]]; then
        echo "Torchtitan unit test directory not found: $titan_test_dir"
        return 1
    fi
    cat > "$conftest_file" << 'EOF'
# Auto-generated conftest for torchtitan-npu patch testing
import pytest

def pytest_configure(config):
    """Import torchtitan_npu to apply NPU patches before running tests."""
    import torchtitan_npu  # noqa: F401
EOF

    # Save original PYTHONPATH and set torchtitan source path
    local saved_pythonpath="$PYTHONPATH"
    export PYTHONPATH="${TITAN_DIR}:${PROJECT_ROOT}:${PYTHONPATH}"

    pytest_args="-v --tb=short --import-mode=importlib"
    # Ignore tests incompatible with NPU environment (ut runs off-device)
    pytest_args="$pytest_args --ignore=tests/unit_tests/test_tokenizer.py"
    pytest_args="$pytest_args --ignore=tests/unit_tests/test_activation_checkpoint.py"
    pytest_args="$pytest_args --ignore=tests/unit_tests/test_download_hf_assets.py"

    # Test target: torchtitan upstream unit tests
    local test_target="tests/unit_tests/"

    # Switch to torchtitan directory (tests use relative paths like ./torchtitan/models/...)
    cd "${TITAN_DIR}"
    echo "Running torchtitan tests from: $(pwd)"
    set +e
    python3 -m pytest $pytest_args $test_target
    local exit_code=$?
    set -e

    # Return to project root
    cd "$PROJECT_ROOT"

    # Cleanup: remove the generated conftest file
    rm -f "$conftest_file"

    # Restore PYTHONPATH
    export PYTHONPATH="$saved_pythonpath"

    if [[ $exit_code -eq 0 ]]; then
        echo "Torchtitan upstream tests passed!"
    elif [[ $exit_code -eq 5 ]]; then
        echo "No torchtitan tests found to run."
    else
        echo "Torchtitan upstream tests failed (exit_code=$exit_code)"
        exit $exit_code
    fi
}



# Run upstream integration tests (torchtitan source with NPU patches)
# Mapping to upstream integration coverage:
# - S-044: model init / basic startup path
# - S-045: dataloader path in upstream smoke suite
# - S-046: forward path
# - S-047: backward path
# - S-048: checkpoint save/load path
# The concrete upstream cases are selected by torchtitan's integration suite
# under tests/integration_tests/.
run_upstream_smoke() {
    echo "Running torchtitan upstream integration tests..."
    mkdir -p "$REPORT_DIR"

    # Ensure torchtitan_npu is installed
    if ! python3 -c "import torchtitan_npu" 2>/dev/null; then
        python3 -m pip install -e .
    fi

    # Check if torchtitan source exists
    if [[ ! -d "$TITAN_DIR" ]]; then
        echo "Cloning torchtitan source..."
        mkdir -p third_party
        git clone --branch $TITAN_VERSION --depth 1 \
            https://gitcode.com/GitHub_Trending/to/torchtitan.git $TITAN_DIR
    fi

    local smoke_log="${REPORT_DIR}/upstream_smoke.log"
    local upstream_output_dir="${REPORT_DIR}/upstream_integration_output"
    local start_time=$(date +%s)
    local titan_test_dir="${TITAN_DIR}/tests/integration_tests"
    local conftest_file="${titan_test_dir}/conftest.py"
    local saved_pythonpath="$PYTHONPATH"

    if [[ ! -d "$titan_test_dir" ]]; then
        echo "Torchtitan integration test directory not found: $titan_test_dir"
        return 1
    fi

    cd "$TITAN_DIR"

    rm -rf "$upstream_output_dir"
    mkdir -p "$upstream_output_dir"

    # Ensure torchtitan_npu patches are applied before upstream integration cases import torchtitan modules.
    cat > "$conftest_file" << 'EOF'
# Auto-generated conftest for torchtitan-npu upstream integration testing
def pytest_configure(config):
    import torchtitan_npu  # noqa: F401
EOF

    export PYTHONPATH="${TITAN_DIR}:${PROJECT_ROOT}:${PYTHONPATH}"

    # Run upstream integration tests with NPU patches applied.
    local cmd=(
        python3 -m tests.integration_tests.run_tests
        "$upstream_output_dir"
        --test_suite features
        --test_name all
        --ngpu 2
    )

    set +e
    "${cmd[@]}" 2>&1 | tee "$smoke_log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    cd "$PROJECT_ROOT"
    rm -f "$conftest_file"
    export PYTHONPATH="$saved_pythonpath"

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "Upstream integration test duration: ${duration}s"

    if [[ $exit_code -eq 0 ]]; then
        echo "Upstream integration tests passed!"
    else
        echo "Upstream smoke failed; stopping smoke pipeline."
        echo "Upstream integration tests failed (exit_code=$exit_code)"
        exit $exit_code
    fi
}



# Core smoke keeps the original semantics of `build.sh -s`:
# run a tiny training configuration through the real training entry.
run_core_smoke() {
    echo "Running core smoke (small-model training path)..."

    cd "$PROJECT_ROOT"
    mkdir -p "$REPORT_DIR"

    local smoke_log="${REPORT_DIR}/smoke_test.log"
    local start_time=$(date +%s)
    local integration_test="${PROJECT_ROOT}/tests/smoke_tests/integration_test.py"
    echo " Verifying torchtitan "
    python -c "import torchtitan; print('torchtitan ok')"

    echo " Done "

    cd "$PROJECT_ROOT"

    echo "Prepared for entering"

    set +e
    timeout $TIMEOUT_SECONDS bash -c "
        python "${integration_test}" "${INTEGRATION_REPORT_DIR}"
    " 2>&1 | tee "$smoke_log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    echo "duration smoke time: ${duration}s"
    if ! analyse_smoke_result "$smoke_log" "$exit_code"; then
        echo "Core smoke failed; stopping smoke pipeline."
        exit 1
    fi
}



analyse_smoke_result() {
    local log_file="$1"
    local exit_code="$2"

    echo "Analysing result of smoke test"
    local has_error=false

    if [[ "$exit_code" -eq 124 ]]; then
        echo "Timeout!"
        has_error=true
    fi

    if grep -qiE "error|exception|traceback" "$log_file" 2>/dev/null; then
        if grep -qiE "NPU error|RuntimeError|OOM|OutOfMemory" "$log_file"; then
            echo "error detected during running"
            grep -iE "NPU error|RuntimeError|OOM|OutOfMemory" "$log_file" | head -10
            has_error=true
        fi
    fi

    if echo "$clean_log" | grep -qiE "loss[^:]*:[^0-9]*(nan|inf)" \
    || echo "$clean_log" | grep -qiE "(nan|inf).*loss"; then
        echo "loss error (NAN/Inf)"
        has_error=true
    fi

    local complete_steps=$(grep -oP "step[:\s]*\K\d+" "$log_file" 2>/dev/null | tail -1 || echo "0")
    if [[ "$complete_steps" -ge "$SMOKE_STEPS" ]]; then
        echo "Completed $complete_steps steps"
    elif [[ "$completed_steps" -gt 0 ]]; then
        echo "Only finished $complete_steps/$SMOKE_STEPS steps"
    else
        echo "No output steps detected"
    fi

    if [[ "$has_error" == "true" ]] || [[ "$exit_code" -ne 0 ]]; then
        return 1
    fi

    return 0
}


# CI tests

run_upstream_ut
pytest -v --tb=short tests/unit_tests
