#!/usr/bin/env bash
# Start full experiments in the current shell; run inside tmux for SSH resilience.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd -- "$PROJECT_ROOT"
[[ -f results/server_setup/activate.sh ]] || {
  printf 'Run scripts/server/setup_smac.sh first.\n' >&2; exit 1;
}
source results/server_setup/activate.sh
python scripts/check_rpg_smac.py --launch --output results/server_setup/game-check.json
exec python scripts/run_rpg_smac_experiments.py --device=cuda "$@"
