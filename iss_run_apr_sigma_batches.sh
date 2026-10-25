#!/bin/bash
#SBATCH --job-name=apr_batch_iss
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%A_%a_apr_batch_iss.out
#SBATCH --error=%A_%a_apr_batch_iss.err

set -eo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || "$#" -ne 2 ]]; then
    echo "Usage: sbatch --array=... iss_run_apr_sigma_batches.sh ABSOLUTE_PLAN_DIRECTORY small|large" >&2
    exit 1
fi

PLAN_DIR="$1"
GROUP="$2"

if [[ ! -f "$PLAN_DIR/plan.json" ]]; then
    echo "ERROR: Missing plan.json in $PLAN_DIR" >&2
    exit 1
fi

if [[ ! -f "$PLAN_DIR/code/apr_sigma_parallel.py" ]]; then
    echo "ERROR: Missing archived runner" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"

# ISS Conda environment
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
echo "ISS APR Scenario Batch"
echo "=========================================="
echo "Array job: ${SLURM_ARRAY_JOB_ID}"
echo "Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Compute node: $(hostname)"
echo "CPUs: $CPU_BUDGET"
echo "Plan: $PLAN_DIR"
echo "Group: $GROUP"
echo "Python: $(command -v python)"
echo "Start: $(date)"
echo "=========================================="

python --version

srun --ntasks=1 \
     --cpus-per-task="$CPU_BUDGET" \
     --cpu-bind=cores \
     python -u "$PLAN_DIR/code/apr_sigma_parallel.py" run \
     --plan "$PLAN_DIR" \
     --group "$GROUP" \
     --task-id "$SLURM_ARRAY_TASK_ID"
