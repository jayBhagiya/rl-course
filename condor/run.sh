#!/usr/bin/env bash
set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR is required}"
: "${DATA_DIR:?DATA_DIR is required}"

UV_VERSION=0.11.30
PYTHON_VERSION=3.10.16
UV_BIN="${DATA_DIR}/tools/uv-${UV_VERSION}/bin/uv"
VENV="${VENV:-${DATA_DIR}/venvs/rl-course-cpu}"
export UV_CACHE_DIR="${DATA_DIR}/cache/uv"
export UV_PYTHON_INSTALL_DIR="${DATA_DIR}/python"
export UV_PROJECT_ENVIRONMENT="${VENV}"
export MPLCONFIGDIR="${DATA_DIR}/cache/matplotlib"
export PYTHONPYCACHEPREFIX="${DATA_DIR}/cache/pycache"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [[ "${1:-}" == "setup" ]]; then
    mkdir -p \
        "${DATA_DIR}/cache/matplotlib" \
        "${DATA_DIR}/cache/pycache" \
        "${DATA_DIR}/cache/uv" \
        "${DATA_DIR}/logs" \
        "${DATA_DIR}/python" \
        "${DATA_DIR}/runs" \
        "$(dirname "${UV_BIN}")" \
        "$(dirname "${VENV}")"
    if [[ ! -x "${UV_BIN}" ]]; then
        python -m pip install --disable-pip-version-check --no-deps \
            --prefix "${DATA_DIR}/tools/uv-${UV_VERSION}" "uv==${UV_VERSION}"
    fi
    "${UV_BIN}" python install "${PYTHON_VERSION}"
    cd "${PROJECT_DIR}"
    "${UV_BIN}" sync --frozen --python "${PYTHON_VERSION}"
    "${VENV}/bin/python" -c \
        'import gymnasium, numpy, racetrackgym, torch; print(torch.__version__, torch.cuda.is_available())'
    exit
fi

PYTHON="${VENV}/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Environment missing at ${VENV}; submit setup.sub first." >&2
    exit 1
fi

cd "${PROJECT_DIR}"
exec "${PYTHON}" -u "$@"
