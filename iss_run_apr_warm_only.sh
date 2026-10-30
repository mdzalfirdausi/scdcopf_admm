#!/bin/bash
#SBATCH --job-name=apr_warm
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%j_apr_warm.out
#SBATCH --error=%j_apr_warm.err

set -eo pipefail

if [[ -z "${SLURM_JOB_ID:-}" || "$#" -ne 3 ]]; then
    echo "Usage: sbatch iss_run_apr_warm_only.sh CASE.xlsx SAVED_LP_DIRECTORY EXPECTED_OBJECTIVE" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
trap 'job_status=$?; echo "Finished: $(date) | Exit code: $job_status"' EXIT

CASE_NAME="$1"
LP_SOURCE="$2"
EXPECTED_OBJECTIVE="$3"

if [[ ! -f run_apr_warm_only.py ]]; then
    echo "ERROR: run_apr_warm_only.py is missing from $PWD." >&2
    exit 1
fi

if [[ ! -d "$LP_SOURCE" ]]; then
    echo "ERROR: Saved LP directory does not exist: $LP_SOURCE" >&2
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
export PYTHONFAULTHANDLER=1

CPU_BUDGET="${SLURM_CPUS_PER_TASK:?Missing CPU allocation}"
CASE_TAG="${CASE_NAME##*/}"
CASE_TAG="${CASE_TAG%.xlsx}"
OUT_DIR="$PWD/output_apr_warm_only_${CASE_TAG}_${SLURM_JOB_ID}"

echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $(hostname)"
echo "CPUs: $CPU_BUDGET"
echo "Case: $CASE_NAME"
echo "Saved LP: $LP_SOURCE"
echo "Expected starting objective: $EXPECTED_OBJECTIVE"
echo "Python: $(command -v python)"
echo "Output: $OUT_DIR"
echo "Start time: $(date)"

python --version

srun --ntasks=1 --cpus-per-task="$CPU_BUDGET" --cpu-bind=cores \
    python -u run_apr_warm_only.py \
    --warm-only \
    --lp-source "$LP_SOURCE" \
    --expected-start-objective "$EXPECTED_OBJECTIVE" \
    --case "$CASE_NAME" \
    --case-dir ../excel_outputs \
    --solver gurobi \
    --threads "$CPU_BUDGET" \
    --gamma 1 \
    --penalty 150000 \
    --islanding skip \
    --mip-time-limit 1800 \
    --mip-gap 1e-7 \
    --apr-tol-mw 1e-4 \
    --feas-tol-mw 1e-4 \
    --output-dir "$OUT_DIR" \
    --tee