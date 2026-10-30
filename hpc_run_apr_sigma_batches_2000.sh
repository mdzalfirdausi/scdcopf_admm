#!/bin/bash
#SBATCH --job-name=apr_2000
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=%A_%a_apr_2000.out
#SBATCH --error=%A_%a_apr_2000.err

set -eo pipefail

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || "$#" -ne 1 ]]; then
    echo "Usage: sbatch --array=0-499%10 hpc_run_apr_sigma_batches_2000.sh PLAN_DIR" >&2
    exit 1
fi

PLAN_DIR="$1"
TASK_ID=$((SLURM_ARRAY_TASK_ID + 1000))

if [[ ! -f "$PLAN_DIR/plan.json" ]]; then
    echo "ERROR: Missing plan.json" >&2
    exit 1
fi

module purge
module load gurobi/13.0.1

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
echo "2000-bus Table B Experiment"
echo "=========================================="
echo "Array job: $SLURM_ARRAY_JOB_ID"
echo "Slurm task: $SLURM_ARRAY_TASK_ID"
echo "Actual plan task: $TASK_ID"
echo "Node: $(hostname)"
echo "CPUs: $CPU_BUDGET"
echo "Start: $(date)"
echo "=========================================="

srun --ntasks=1 \
     --cpus-per-task="$CPU_BUDGET" \
     --cpu-bind=cores \
     python -u "$PLAN_DIR/code/apr_sigma_parallel.py" run \
     --plan "$PLAN_DIR" \
     --group large \
     --task-id "$TASK_ID"
