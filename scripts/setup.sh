#!/usr/bin/env bash
# Run from any directory. Activate the resulting environment separately.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="${PYTHON_BIN:-python3}"
"$python_command" -m venv "$project_root/.venv"
if ! "$project_root/.venv/bin/python" -c 'import torch' >/dev/null 2>&1; then
  "$project_root/.venv/bin/python" -m pip install --no-cache-dir \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}" 'torch>=2.2,<3'
fi
"$project_root/.venv/bin/python" -m pip install --no-cache-dir -e "${project_root}[dev]"
"$project_root/.venv/bin/python" -m pip check
echo "Activate with: source $project_root/.venv/bin/activate"
