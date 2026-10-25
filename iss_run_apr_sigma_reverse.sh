#!/bin/bash
#SBATCH --job-name=apr_reverse_iss
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%j_apr_reverse_iss.out
#SBATCH --error=%j_apr_reverse_iss.err

set -eo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit using sbatch." >&2
    exit 1
fi

if [[ "$#" -ne 3 ]]; then
    echo "Usage: sbatch iss_run_apr_sigma_reverse.sh PLAN_DIR START_ID END_ID" >&2
    exit 1
fi

PLAN_DIR="$1"
START_ID="$2"
END_ID="$3"

if [[ ! -f "$PLAN_DIR/plan.json" ]]; then
    echo "ERROR: Missing plan.json" >&2
    exit 1
fi

if [[ ! -f "$PLAN_DIR/code/apr_sigma_parallel.py" ]]; then
    echo "ERROR: Missing archived runner" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch

set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

CPU_BUDGET="${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"

echo "=========================================="
echo "ISS Reverse APR Scenario Execution"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $CPU_BUDGET"
echo "Plan: $PLAN_DIR"
echo "Tasks: $START_ID down to $END_ID"
echo "Start time: $(date)"
echo "=========================================="

python --version

for TASK_ID in $(seq "$START_ID" -1 "$END_ID"); do

    echo ""
    echo "=========================================="
    echo "Starting task $TASK_ID"
    echo "Time: $(date)"
    echo "=========================================="

    srun --ntasks=1 \
         --cpus-per-task="$CPU_BUDGET" \
         --cpu-bind=cores \
         python -u "$PLAN_DIR/code/apr_sigma_parallel.py" run \
         --plan "$PLAN_DIR" \
         --group small \
         --task-id "$TASK_ID"

    echo "Completed task $TASK_ID at $(date)"

done

echo "=========================================="
echo "All assigned ISS tasks completed."
echo "Finished: $(date)"
echo "=========================================="
