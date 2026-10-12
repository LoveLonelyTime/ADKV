#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

# Prerequisite: wtzp_wts_fig.sh
python "$PROJECT_ROOT/tools/trace_fig.py"