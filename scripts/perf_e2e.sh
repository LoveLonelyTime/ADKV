#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python "$PROJECT_ROOT/tools/perf_e2e.py" --batch_size 1
python "$PROJECT_ROOT/tools/perf_e2e.py" --batch_size 2
python "$PROJECT_ROOT/tools/perf_e2e.py" --batch_size 4
python "$PROJECT_ROOT/tools/perf_e2e.py" --batch_size 8
python "$PROJECT_ROOT/tools/perf_e2e.py" --batch_size 16
python "$PROJECT_ROOT/tools/perf_e2e.py" --batch_size 32
python "$PROJECT_ROOT/tools/perf_e2e.py" --batch_size 64
python "$PROJECT_ROOT/tools/perf_e2e.py" --batch_size 128
python "$PROJECT_ROOT/tools/perf_e2e.py" --batch_size 256
