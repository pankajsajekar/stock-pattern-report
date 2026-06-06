#!/usr/bin/env bash
# Convenience runner: sets up venv on first run, then generates the report.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtualenv + installing deps (first run only)..."
  python3 -m venv .venv
  .venv/bin/pip install --quiet --upgrade pip
  .venv/bin/pip install --quiet yfinance pandas numpy scipy mplfinance jinja2
fi

# Pass any args straight through, e.g.:  ./run.sh --period 1y
.venv/bin/python analyze.py "$@"
