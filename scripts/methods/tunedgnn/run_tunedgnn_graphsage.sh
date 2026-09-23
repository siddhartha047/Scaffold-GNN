#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TUNEDGNN_MODEL="sage"
export BASELINE_METHOD_NAME="tunedgnn-graphsage"
exec "${SCRIPT_DIR}/run_tunedgnn.sh" "$@"
