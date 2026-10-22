#!/bin/bash
#SBATCH --job-name=apr_bisection
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%j_apr_bisection.out
#SBATCH --error=%j_apr_bisection.err

set -eo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit with sbatch, not bash." >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"
trap 'job_status=$?; echo "Finished: $(date) | Exit code: $job_status"' EXIT

if [[ ! -f check_apr_eta_limits.py ]]; then
    echo "ERROR: check_apr_eta_limits.py is missing from $PWD." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
set -u

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CPU_BUDGET="${SLURM_CPUS_PER_TASK:?Missing Slurm CPU allocation}"
OUT_DIR="$PWD/output_apr_bisection_case300_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $(hostname)"
echo "CPUs: $CPU_BUDGET"
echo "Environment: pytorch"
echo "Python: $(command -v python)"
echo "Output: $OUT_DIR"
echo "Start time: $(date)"

python --version

srun --ntasks=1 --cpus-per-task="$CPU_BUDGET" --cpu-bind=cores \
    python -u check_apr_eta_limits.py \
    --case pglib_opf_case1354_pegase.xlsx \
    --case-dir ../excel_outputs \
    --solver gurobi \
    --threads "$CPU_BUDGET" \
    --gamma 1.0 \
    --thermal-mode soft \
    --weight-bisection \
    --slack-normalization none \
    --weight-start 1 \
    --weight-tol 0.001 \
    --weight-max-solves 40 \
    --islanding skip \
    --output-dir "$OUT_DIR"