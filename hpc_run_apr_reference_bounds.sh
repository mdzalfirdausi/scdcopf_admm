#!/bin/bash
#SBATCH --job-name=apr_bound
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%j_apr_bound.out
#SBATCH --error=%j_apr_bound.err

set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit with sbatch from your project directory." >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"
trap 'job_status=$?; echo "Finished: $(date) | Exit code: $job_status"' EXIT
if [[ ! -f fill_apr_reference_bounds.py || ! -f apr_bound_core.py ]]; then
    echo "ERROR: Extract the reference-bound Python scripts into your project directory." >&2
    exit 1
fi
EXTRA_ARGS=("$@")
HAS_OUTPUT=0
for argument in "$@"; do
    case "$argument" in --output-dir|--output-dir=*) HAS_OUTPUT=1 ;; esac
done
if [[ "$HAS_OUTPUT" -eq 0 ]]; then
    EXTRA_ARGS+=(--output-dir "$PWD/output_apr_reference_${SLURM_JOB_ID}")
fi
module purge
module load gurobi/13.0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
gurobi_cl --license
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
CPU_BUDGET="${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $(hostname)"
echo "CPUs: $CPU_BUDGET"
echo "Start time: $(date)"
python --version
srun --ntasks=1 --cpus-per-task="$CPU_BUDGET" --cpu-bind=cores \
    python -u fill_apr_reference_bounds.py --solver gurobi --threads "$CPU_BUDGET" "${EXTRA_ARGS[@]}"
