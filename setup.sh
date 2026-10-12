#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$PWD/.venv/bin:$PATH"
if [[ "${1:-}" == "--help" || "${1:-}" == "--check" ]]; then
  python_bin="${STRATA_PYTHON:-python3.12}"
  [[ ! -x .venv/bin/python ]] || python_bin=.venv/bin/python
  exec "$python_bin" tools/install.py "$@"
fi
python_bin="${STRATA_PYTHON:-python3.12}"
"$python_bin" -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ required"'
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install -r requirements.txt -r requirements-pack.txt
.venv/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
exec .venv/bin/python tools/install.py "$@"
