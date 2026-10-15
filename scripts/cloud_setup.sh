#!/usr/bin/env bash
# Set up a Claude Code cloud VM (Ubuntu, CPU only) for FIM-GAT v2: a virtualenv at .venv with CPU PyTorch and the
# library versions tested on the development laptop (requirements.txt). Idempotent; rerun it in every new session.
#
#   bash scripts/cloud_setup.sh && source .venv/bin/activate
#
# Network: PyPI is allowed by default. CPU PyTorch wheels come from download.pytorch.org when the environment allows
# that domain (smaller, faster); otherwise PyPI's default torch wheel is used, which works on a CPU VM but downloads
# the CUDA libraries too. Dataset builds need snap.stanford.edu allowed as well (docs/cloud_runbook.md).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
"$PY" --version

make_venv() {
    # plain venv first; Ubuntu images without python3-venv need one of the fallbacks
    "$PY" -m venv .venv 2>/dev/null && return 0
    rm -rf .venv
    if command -v uv >/dev/null 2>&1 && uv venv --seed --python "$PY" .venv; then return 0; fi
    local SUDO=""
    [ "$(id -u)" = 0 ] || SUDO="sudo -n"
    if command -v apt-get >/dev/null 2>&1 && $SUDO apt-get update -qq \
        && $SUDO apt-get install -y -qq "python3-venv" >/dev/null && "$PY" -m venv .venv; then return 0; fi
    rm -rf .venv
    "$PY" -m pip install --quiet --user --break-system-packages virtualenv 2>/dev/null \
        || "$PY" -m pip install --quiet --user virtualenv
    "$PY" -m virtualenv .venv
}
if [ ! -x .venv/bin/python ]; then
    make_venv
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

if ! python -c "import torch" 2>/dev/null; then
    python -m pip install --quiet "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu \
        || python -m pip install --quiet "torch==2.6.0"
fi
python -m pip install --quiet "torch_geometric==2.7.0" "numba==0.63.1" "numpy==2.3.5" "scipy==1.17.0" \
    "scikit-learn==1.8.0" "pandas==3.0.0" "networkx==3.6.1" "psutil==7.2.1" "matplotlib==3.10.8" "pytest==9.0.3"

python - <<'EOF'
import os, numba, numpy, pandas, torch, torch_geometric
print(f"python ok: torch {torch.__version__} (cuda {torch.cuda.is_available()}), torch_geometric "
      f"{torch_geometric.__version__}, numba {numba.__version__} ({numba.config.NUMBA_NUM_THREADS} threads), "
      f"numpy {numpy.__version__}, pandas {pandas.__version__}; {os.cpu_count()} CPUs")
EOF
