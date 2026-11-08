#!/usr/bin/env bash
# Formal training defaults to CUDA; CPU validation must be selected explicitly.
set -euo pipefail
device=cuda
case "${1:---cuda}" in
  --cuda) device=cuda ;;
  --cpu) device=cpu ;;
  --help|-h)
    echo 'Usage: [TORCH_INDEX_URL=<official CUDA wheel index>] bash scripts/setup.sh [--cuda|--cpu]'
    echo 'Default: --cuda. A new CUDA environment needs the index from https://pytorch.org/get-started/locally/.'
    exit 0 ;;
  *) echo "Unknown option: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  echo 'Pass only --cuda or --cpu; configure the wheel index with TORCH_INDEX_URL.' >&2
  exit 2
fi
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_command="${PYTHON_BIN:-python3}"
"$python_command" -m venv "$project_root/.venv"
venv_python="$project_root/.venv/bin/python"
installed_cuda="$("$venv_python" -c 'import torch; print(torch.version.cuda or "cpu")' 2>/dev/null || true)"
torch_index="${TORCH_INDEX_URL:-}"
install_torch=0
if [[ "$device" == cuda ]]; then
  if [[ -n "$torch_index" ]]; then
    if [[ ! "$torch_index" =~ ^https://download\.pytorch\.org/whl/cu[0-9]+/?$ ]]; then
      echo 'CUDA mode requires an official CUDA wheel index such as https://download.pytorch.org/whl/cu128.' >&2
      exit 1
    fi
    installed_tag="cu${installed_cuda//./}"
    [[ "${torch_index%/}" != "https://download.pytorch.org/whl/$installed_tag" ]] && install_torch=1
  elif [[ -z "$installed_cuda" || "$installed_cuda" == cpu ]]; then
    echo 'CUDA mode cannot reuse CPU torch. Set TORCH_INDEX_URL to the CUDA index selected at https://pytorch.org/get-started/locally/.' >&2
    echo 'For local CPU validation instead, run: bash scripts/setup.sh --cpu' >&2
    exit 1
  fi
else
  torch_index="${torch_index:-https://download.pytorch.org/whl/cpu}"
  if [[ "${torch_index%/}" != https://download.pytorch.org/whl/cpu ]]; then
    echo 'CPU mode requires the CPU wheel index; unset TORCH_INDEX_URL or set it to https://download.pytorch.org/whl/cpu.' >&2
    exit 1
  fi
  [[ "$installed_cuda" != cpu ]] && install_torch=1
fi
if (( install_torch )); then
  # Force replacement when changing flavor, even if CPU torch has a newer version.
  "$venv_python" -m pip install --no-cache-dir --force-reinstall \
    --index-url "$torch_index" 'torch>=2.2,<3'
fi
"$venv_python" -m pip install --no-cache-dir -e "${project_root}[dev]"
"$venv_python" -m pip check
"$venv_python" - "$device" <<'PY'
import sys
import torch
requested = sys.argv[1]
print(f"torch={torch.__version__}; wheel CUDA={torch.version.cuda}; CUDA available={torch.cuda.is_available()}")
if requested == "cuda":
    if torch.version.cuda is None or not torch.cuda.is_available():
        raise SystemExit("CUDA setup is not ready: check the NVIDIA GPU, driver, container GPU access and selected wheel. CPU fallback is disabled.")
    print("GPU:", torch.cuda.get_device_name(0))
elif torch.version.cuda is not None:
    raise SystemExit("CPU mode unexpectedly retained a CUDA wheel.")
PY
echo "Activate with: source $project_root/.venv/bin/activate"
