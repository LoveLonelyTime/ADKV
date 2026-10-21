#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m adkv.calib_collector \
    --layer=0-31 \
    --clip_len=16000 \
    --shuffle \
    --n_samples=512 \
    --max_length=4096 \
    --batch_size=16 \
    --layers_per_chunk=4 \
    --output=calib/calib_llama.pt
