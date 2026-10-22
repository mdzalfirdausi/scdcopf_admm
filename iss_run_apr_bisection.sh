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

CASE_NAME=""
CASE_DIR="../excel_outputs"
OUT_DIR=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --case|--case-dir|--output-dir)
            if [[ "$#" -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "ERROR: $1 requires a value." >&2
                exit 1
            fi
            case "$1" in
                --case) CASE_NAME="$2" ;;
                --case-dir) CASE_DIR="$2" ;;
                --output-dir) OUT_DIR="$2" ;;
            esac
            shift 2
            ;;
        *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CASE_NAME" ]]; then
    echo "Usage: sbatch iss_run_apr_bisection.sh --case CASE.xlsx [--case-dir PATH] [--output-dir PATH]" >&2
    exit 1
fi

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
CASE_TAG="${CASE_NAME##*/}"
CASE_TAG="${CASE_TAG%.xlsx}"
if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$PWD/output_apr_bisection_${CASE_TAG}_${SLURM_JOB_ID}"
fi
mkdir -p "$OUT_DIR"

echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $(hostname)"
echo "CPUs: $CPU_BUDGET"
echo "Case: $CASE_NAME"
echo "Case directory: $CASE_DIR"
echo "Environment: pytorch"
echo "Python: $(command -v python)"
echo "Output: $OUT_DIR"
echo "Start time: $(date)"

python --version

srun --ntasks=1 --cpus-per-task="$CPU_BUDGET" --cpu-bind=cores \
    python -u check_apr_eta_limits.py \
    --case "$CASE_NAME" \
    --case-dir "$CASE_DIR" \
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
