#!/bin/bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

python -m adkv.train.adkv_train_ray \
    --atten=llama \
    --num_hidden_states=4096 \
    --layers=0-31 \
    --epochs=100 \
    --batch_size=4 \
    --truncation=256 \
    --learning_rate=1e-2 \
    --bits=4