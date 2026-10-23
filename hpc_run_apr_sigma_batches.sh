#!/bin/bash
#SBATCH --job-name=apr_batch
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%A_%a_apr_batch.out
#SBATCH --error=%A_%a_apr_batch.err

set -eo pipefail
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || "$#" -ne 2 ]]; then
    echo "Usage: sbatch --array=... hpc_run_apr_sigma_batches.sh ABSOLUTE_PLAN_DIRECTORY small|large" >&2
    exit 1
fi
PLAN_DIR="$1"
GROUP="$2"
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
CPU_BUDGET="${SLURM_CPUS_PER_TASK:?Missing allocation}"
echo "Array: ${SLURM_ARRAY_JOB_ID}; task: $SLURM_ARRAY_TASK_ID; node: $(hostname); CPUs: $CPU_BUDGET"
python --version
srun --ntasks=1 --cpus-per-task="$CPU_BUDGET" --cpu-bind=cores \
    python -u "$PLAN_DIR/code/apr_sigma_parallel.py" run \
    --plan "$PLAN_DIR" --group "$GROUP" --task-id "$SLURM_ARRAY_TASK_ID"
