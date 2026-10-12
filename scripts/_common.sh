_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$_COMMON_DIR")"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
