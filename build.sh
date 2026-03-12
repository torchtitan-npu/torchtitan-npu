# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

show_help() {
    cat << EOF
torchtitan-npu build script
sh build.sh [options]

Build:
    -b, --build                       Build whl package
    -c, --clean                       Clean build artifacts
    -i, --install                     Perform developer installation

Test:
    -u, --ut                          Run unit tests
    -s, --smoke                       Run smoke tests

Examples:
    sh build.sh -a                    # Run build, unit tests, and smoke tests
    sh build.sh -f pr_list.txt -u     # Incremental unit tests
EOF
    exit 0
}

set -e

# Global variable
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
OUTPUT_DIR="${PROJECT_ROOT}/output"
REPORT_DIR="${PROJECT_ROOT}/test_reports"
TITAN_VERSION="v0.2.1"
TITAN_DIR="third_party/torchtitan"

# Default configuration
NGPU=2
SMOKE_STEP=10
SMOKE_CONFIG="${PROJECT_ROOT}/tests/smoke_tests/smoke_test.toml"
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-300}

# Colors and logging
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Python 3.10 paths
PYTHON310_INSTALL_DIR="/usr/local/python3.10.18"
PYTHON310_LINK_DIR="/usr/local/python/python310"

# Setup Python 3.10 environment
setup_python310() {
    local python_bin_dir="${PYTHON310_LINK_DIR}/bin"

    # Check if python3.10 exists and has required modules
    local need_install=false
    if [[ ! -x "${python_bin_dir}/python3" ]]; then
        log_info "Python 3.10 not found at ${python_bin_dir}"
        need_install=true
    elif ! "${python_bin_dir}/python3" -c "import sqlite3, bz2, lzma" 2>/dev/null; then
        log_info "Python 3.10 missing required modules (sqlite3, bz2, lzma)"
        need_install=true
    fi

    # Install Python 3.10 if needed
    if [[ "$need_install" == "true" ]]; then
        log_info "Installing Python 3.10..."

        echo "  Installing build dependencies "
        apt update
        apt install -y build-essential wget libbz2-dev liblzma-dev libsqlite3-dev libffi-dev libreadline-dev libncurses5-dev zlib1g-dev libssl-dev

        echo "  Downloading Python 3.10 source "
        cd /tmp
        wget --no-check-certificate https://www.python.org/ftp/python/3.10.18/Python-3.10.18.tgz
        tar -xzf Python-3.10.18.tgz
        cd Python-3.10.18

        echo "  Compiling and installing "
        ./configure --prefix="$PYTHON310_INSTALL_DIR"
        make -j$(nproc)
        make install

        echo "  Creating symbolic links "
        rm -rf "$PYTHON310_LINK_DIR"
        ln -sf "$PYTHON310_INSTALL_DIR" "$PYTHON310_LINK_DIR"

        echo "  Verifying "
        "${python_bin_dir}/python3" -c "import sqlite3, bz2, lzma; print('Modules complete: ok')"
        "${python_bin_dir}/python3" --version

        cd "$PROJECT_ROOT"
    fi

    # Create symlinks for python/pip (if python3/pip3 exist but python/pip don't)
    ln -sf "${python_bin_dir}/python3" "${python_bin_dir}/python" 2>/dev/null || true
    ln -sf "${python_bin_dir}/pip3" "${python_bin_dir}/pip" 2>/dev/null || true

    # Add Python bin dir to PATH with highest priority
    export PATH="${python_bin_dir}:$PATH"

    log_info "Python 3.10 ready: $(which python)"
}

# Setup Python 3.10 environment
setup_python310

# Operation sign
DO_BUILD=false
DO_UNIT_TEST=false
DO_SMOKE_TEST=false
DO_CLEAN=false
DO_INSTALL=false
PR_FILELIST=""

parse_args() {
    if [[ $# -eq 0 ]]; then
        DO_BUILD=true
        return
    fi

    while [[ $# -gt 0 ]]; do
        case $1 in
            -b|--build)
                DO_BUILD=true
                shift
                ;;
            -c|--clean)
                DO_CLEAN=true
                shift
                ;;
            -i|--install)
                DO_INSTALL=true
                shift
                ;;
            -u|--ut)
                DO_UNIT_TEST=true
                shift
                ;;
            -s|--smoke)
                DO_SMOKE_TEST=true
                shift
                ;;
            -a|--all)
                DO_BUILD=true
                DO_UNIT_TEST=true
                DO_SMOKE_TEST=true
                shift
                ;;
            -f|--filelist)
                PR_FILELIST="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                ;;
            *)
                log_error "Unknown param"
                exit 1
                ;;
        esac
    done
}


# Environment Inspection
check_python_env() {
    log_info "Checking Python..."
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not installed"
        exit 1
    fi
}

check_dependencies() {
    log_info "Checking dependencies..."

    python3 -m pip install --upgrade pip

    # Uninstall triton before installing triton_ascend (they conflict)
    if python3 -c "import triton" 2>/dev/null; then
        log_info "Uninstalling triton (conflicts with triton_ascend)..."
        python3 -m pip uninstall -y triton
    fi

    # Use CI-specific requirements if available, otherwise use default
    local req_file="${PROJECT_ROOT}/requirements_ci.txt"
    if [[ ! -f "$req_file" ]]; then
        req_file="${PROJECT_ROOT}/requirements.txt"
    fi

    if [[ -f "$req_file" ]]; then
        log_info "Installing dependencies from $(basename $req_file)..."
        python3 -m pip install -r "$req_file" | grep -v "Requirement already satisfied"
    fi
    python3 -m pip list
}

check_environment() {
    check_python_env
    check_dependencies
}

# do_clean
do_clean() {
    cd "$PROJECT_ROOT"

    rm -rf build/ dist/ *.egg-info/ output/
    rm -rf .pytest_cache/ .coverage htmlcov/
    rm -rf test_reports/

    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    log_info "Cleaning complete"
}

# do_build
do_build() {
    log_info "Building whl"

    cd "$PROJECT_ROOT"
    mkdir -p "$OUTPUT_DIR"

    python -m pip install --quiet build wheel setuptools

    python -m build --wheel --outdir "$OUTPUT_DIR"

    if [[ $? -eq 0 ]]; then
        log_info "Build success!"
        ls -la "$OUTPUT_DIR"/*.whl 2>/dev/null || log_error "WHL file not found"
    else
        log_error "Build failure!"
        exit 1
    fi
}

# Development mode install
do_install() {
    log_info "Development Mode Installation"

    cd "$PROJECT_ROOT"
    python3 -m pip install -e .

    log_info "Installation success!"
}

# Run torchtitan upstream unit tests (with NPU patches applied)
run_torchtitan_tests() {
    log_info "Running torchtitan upstream unit tests..."

    mkdir -p "$REPORT_DIR"

    # Ensure torchtitan_npu is installed (applies patches on import)
    if ! python3 -c "import torchtitan_npu" 2>/dev/null; then
        python3 -m pip install -e .
    fi

    # Clone torchtitan source if not exists
    if [ ! -d "$TITAN_DIR" ]; then
        log_info "Cloning torchtitan source..."
        mkdir -p third_party
        git clone --branch $TITAN_VERSION --depth 1 \
            https://gitcode.com/GitHub_Trending/to/torchtitan.git $TITAN_DIR
    fi

    # Create conftest.py in torchtitan test dir to ensure import torchtitan_npu before each test
    local titan_test_dir="${TITAN_DIR}/tests/unit_tests"
    local conftest_file="${titan_test_dir}/conftest.py"
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

    local pytest_args="-v --tb=short --import-mode=importlib"
    pytest_args="$pytest_args --html=${PROJECT_ROOT}/${REPORT_DIR}/torchtitan_ut_report.html --self-contained-html"

    # Ignore tests incompatible with NPU environment (ut runs off-device)
    pytest_args="$pytest_args --ignore=tests/unit_tests/test_tokenizer.py"
    pytest_args="$pytest_args --ignore=tests/unit_tests/test_activation_checkpoint.py"
    pytest_args="$pytest_args --ignore=tests/unit_tests/test_download_hf_assets.py"

    # Test target: torchtitan upstream unit tests
    local test_target="tests/unit_tests/"

    # Switch to torchtitan directory (tests use relative paths like ./torchtitan/models/...)
    cd "${TITAN_DIR}"
    log_info "Running torchtitan tests from: $(pwd)"
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
        log_info "Torchtitan upstream tests passed!"
    elif [[ $exit_code -eq 5 ]]; then
        log_info "No torchtitan tests found to run."
    else
        log_error "Torchtitan upstream tests failed (exit_code=$exit_code)"
        return $exit_code
    fi
}

# Run torchtitan-npu local unit tests
# Optional param: test_file_list (for incremental testing)
run_npu_tests() {
    log_info "Running torchtitan-npu local unit tests..."

    mkdir -p "$REPORT_DIR"

    # Ensure torchtitan_npu is installed
    if ! python3 -c "import torchtitan_npu" 2>/dev/null; then
        python3 -m pip install -e .
    fi

    local pytest_args="-v --tb=short --confcutdir=tests --import-mode=importlib"
    pytest_args="$pytest_args --html=${REPORT_DIR}/npu_ut_report.html --self-contained-html"
    pytest_args="$pytest_args --cov=torchtitan_npu --cov-report=html:${REPORT_DIR}/coverage"
    pytest_args="$pytest_args --cov-report=term-missing"

    # Test target: local npu unit tests
    local test_target="tests/unit_tests/"

    # Support incremental testing via PR_FILELIST
    if [[ -n "$PR_FILELIST" && -f "$PR_FILELIST" ]]; then
        log_info "Filtering tests based on PR list..."
        local filtered_tests=$(get_affected_tests "$PR_FILELIST")
        if [[ -n "$filtered_tests" ]]; then
            test_target="$filtered_tests"
        else
            log_info "No relevant local test cases to run based on PR list."
            return 0
        fi
    fi

    if [[ ! -d "tests/unit_tests" ]]; then
        log_error "Local unit_tests directory not found"
        return 1
    fi

    log_info "Running NPU tests from: $test_target"
    set +e
    python3 -m pytest $pytest_args $test_target
    local exit_code=$?
    set -e

    if [[ $exit_code -eq 0 ]]; then
        log_info "Torchtitan-npu local tests passed!"
    elif [[ $exit_code -eq 5 ]]; then
        log_info "No local tests found to run."
    else
        log_error "Torchtitan-npu local tests failed (exit_code=$exit_code)"
        return $exit_code
    fi
}

# Unit test
do_unit_test() {
    log_info "Start unit tests..."

    cd "$PROJECT_ROOT"

    python3 -m pip install pytest pytest-html pytest-cov

    log_info "listing dependencies..."
    python3 -m pip list

    local overall_exit_code=0

    # Incremental test mode: only run local tests filtered by PR_FILELIST
    if [[ -n "$PR_FILELIST" && -f "$PR_FILELIST" ]]; then
        log_info "Incremental test mode enabled (PR_FILELIST: $PR_FILELIST)"
        run_npu_tests
        local npu_exit=$?
        if [[ $npu_exit -ne 0 && $npu_exit -ne 5 ]]; then
            log_error "Incremental tests failed!"
            exit 1
        fi
        log_info "Incremental tests completed!"
        return 0
    fi

    # Full test mode: run both test phases

    # Phase 1: Run torchtitan upstream tests (with NPU patches)
    run_torchtitan_tests
    local titan_exit=$?
    if [[ $titan_exit -ne 0 && $titan_exit -ne 5 ]]; then
        overall_exit_code=1
    fi

    # Phase 2: Run torchtitan-npu local tests
    run_npu_tests
    local npu_exit=$?
    if [[ $npu_exit -ne 0 && $npu_exit -ne 5 ]]; then
        overall_exit_code=1
    fi

    if [[ $overall_exit_code -eq 0 ]]; then
        log_info "All unit tests passed!"
    else
        log_error "Some unit tests failed!"
        exit 1
    fi
}

# only add increment tests
get_affected_tests() {
    local filelist="$1"
    local test_files=""

    while IFS= read -r changed_file || [[ -n "$changed_file" ]]; do
        [[ -z "$changed_file" || "$changed_file" == \#* ]] && continue

        if [[ "$changed_file" == torchtitan_npu/*.py ]]; then
            local module_name=$(basename "changed_file" .py)
            local test_file="tests/test_${module_name}.py"
            if [[ -f "$test_file" ]]; then
                test_files="$test_files $test_file"
            fi
        fi

        if [[ "$changed_file" == tests/*.py ]]; then
            if [[ -f "changed_file" ]]; then
                test_files="$test_files $test_file"
            fi
        fi
    done < "$filelist"

    echo "$test_files" | tr ' ' '\n' | sort -u | tr '\n' ' '
}

do_smoke_test() {
    log_info "Smoke test start..."

    cd "$PROJECT_ROOT"
    mkdir -p "$REPORT_DIR"

    local smoke_log="${REPORT_DIR}/smoke_test.log"
    local start_time=$(date +%s)

    echo " Verifying torchtitan "
    python -c "import torchtitan; print('torchtitan ok')"

    echo " Done "

    cd "$PROJECT_ROOT"
    chmod +x ./run_train.sh

    log_info "Prepared for entering"

    set +e 
    timeout $TIMEOUT_SECONDS bash -c "
        export NGPU=$NGPU
        export CONFIG_FILE="${SMOKE_CONFIG}"
        ./run_train.sh
    " 2>&1 | tee "$smoke_log"
    local exit_code=${PIPESTATUS[0]}
    set -e

    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    log_info "duration smoke time: ${duration}s"
    analyse_smoke_result "$smoke_log" "$exit_code"
}

analyse_smoke_result() {
    local log_file="$1"
    local exit_code="$2"

    log_info "Analysing result of smoke test"
    local has_error=false

    if [[ "$exit_code" -eq 124 ]]; then
        log_error "Timeout!"
        has_error=true
    fi

    if grep -qiE "error|exception|traceback" "$log_file" 2>/dev/null; then
        if grep -qiE "NPU error|RuntimeError|OOM|OutOfMemory" "$log_file"; then
            log_error "error detected during running"
            grep -iE "NPU error|RuntimeError|OOM|OutOfMemory" "$log_file" | head -10
            has_error=true
        fi
    fi

    if echo "$clean_log" | grep -qiE "loss[^:]*:[^0-9]*(nan|inf)" \
    || echo "$clean_log" | grep -qiE "(nan|inf).*loss"; then
        log_error "loss error (NAN/Inf)"
        has_error=true
    fi

    local complete_steps=$(grep -oP "step[:\s]*\K\d+" "$log_file" 2>/dev/null | tail -1 || echo "0")
    if [[ "$completed_steps" -ge "$SMOKE_STEPS" ]]; then
        log_info "Completed $complete_steps steps"
    elif [[ "$completed_steps" -gt 0 ]]; then
        log_error "Only finished $complete_steps/$SMOKE_STEPS steps"
    else
        log_error "No output steps detected"
    fi

    if [[ "$has_error" == "true" ]] || [[ "$exit_code" -ne 0 ]]; then
        log_error "smoke test failed"
        exit 1
    else
        log_info "smoke test pass"
        exit 0
    fi
}

main() {
    parse_args "$@"

    log_info "Torchtitan-npu build"

    if [[ "$DO_BUILD" == "true" ]] || [[ "$DO_UNIT_TEST" == "true" ]] || [[ "$DO_SMOKE_TEST" == "true" ]]; then 
        check_environment
    fi

    if [[ "$DO_CLEAN" == "true" ]]; then
        do_clean
    fi

    if [[ "$DO_BUILD" == "true" ]]; then
        do_build
    fi

    if [[ "$DO_INSTALL" == "true" ]]; then
        do_install
    fi

    if [[ "$DO_UNIT_TEST" == "true" ]]; then
        do_unit_test
    fi

    if [[ "$DO_SMOKE_TEST" == "true" ]]; then
        do_smoke_test
    fi
}

main "$@"