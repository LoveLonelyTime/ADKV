#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python "$PROJECT_ROOT/tools/perf_kernel.py"
