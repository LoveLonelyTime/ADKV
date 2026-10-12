#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python "$PROJECT_ROOT/tools/wtzp_wts_fig.py" --layers 0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30