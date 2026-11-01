#!/bin/bash
set -euo pipefail

PLAN_DIR="${1:?Missing plan directory}"
PLAN_TASK_ID=$((SLURM_ARRAY_TASK_ID + 1000))

echo "Slurm array task: $SLURM_ARRAY_TASK_ID"
echo "Mapped plan task: $PLAN_TASK_ID"

exec env SLURM_ARRAY_TASK_ID="$PLAN_TASK_ID" \
    bash "$PLAN_DIR/code/hpc_run_apr_sigma_batches.sh" \
    "$PLAN_DIR" large
