#!/bin/bash
#SBATCH --job-name=iss_line_scopf
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=iss_line_scopf_%j.out
#SBATCH --error=iss_line_scopf_%j.err

# Submit from the directory containing benchmark_line_scopf.py.
# sbatch iss_run_line_scopf.sh --case CASE.xlsx --thermal-mode hard --repeat 1
# No array and no GPU: one CPU job compares the joint LP and parallel ADMM.
# Optional node selection belongs on the sbatch command: --nodelist=issnode1.
# Default resources are pilot estimates; inspect MaxRSS after the first run.

set -eo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit with sbatch, not bash, from your project directory." >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"
trap 'job_status=$?; echo "Finished: $(date) | Exit code: $job_status"' EXIT

CASE_NAME=""
CASE_DIR="../excel_outputs"
THERMAL_MODE="hard"
ISLANDING="error"
REPEATS=1

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --case|--case-dir|--thermal-mode|--islanding|--repeat)
            if [[ "$#" -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "ERROR: $1 requires a value." >&2
                exit 1
            fi
            case "$1" in
                --case) CASE_NAME="$2" ;;
                --case-dir) CASE_DIR="$2" ;;
                --thermal-mode) THERMAL_MODE="$2" ;;
                --islanding) ISLANDING="$2" ;;
                --repeat) REPEATS="$2" ;;
            esac
            shift 2
            ;;
        *) echo "ERROR: Unknown parameter: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$CASE_NAME" ]]; then
    echo "Usage: sbatch iss_run_line_scopf.sh --case CASE.xlsx [--thermal-mode hard|soft] [--islanding error|skip] [--repeat N] [--case-dir PATH]" >&2
    exit 1
fi
if [[ "$THERMAL_MODE" != hard && "$THERMAL_MODE" != soft ]]; then
    echo "ERROR: --thermal-mode must be hard or soft." >&2
    exit 1
fi
if [[ "$ISLANDING" != error && "$ISLANDING" != skip ]]; then
    echo "ERROR: --islanding must be error or skip." >&2
    exit 1
fi
if [[ ! "$REPEATS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --repeat must be a positive integer." >&2
    exit 1
fi
if [[ ! -f benchmark_line_scopf.py ]]; then
    echo "ERROR: benchmark_line_scopf.py is missing from $PWD." >&2
    exit 1
fi

# Use the environment initialization from your ISS example.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
set -u

# Each ADMM worker receives one Gurobi solver thread. Do not also allow each
# worker to create a full 16-thread BLAS/OpenMP team.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

CPU_BUDGET="${SLURM_CPUS_PER_TASK:?Missing Slurm CPU allocation}"
CASE_TAG="${CASE_NAME##*/}"
CASE_TAG="${CASE_TAG%.xlsx}"
OUT_DIR="$PWD/output_iss/${CASE_TAG}_${THERMAL_MODE}_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Batch node: $(hostname)"
echo "CPUs: $CPU_BUDGET"
echo "Case: $CASE_NAME"
echo "Mode: $THERMAL_MODE; islanding policy: $ISLANDING; repeats: $REPEATS"
echo "Environment: pytorch"
echo "Python: $(command -v python)"
echo "Output: $OUT_DIR"
echo "Start time: $(date)"
echo "============================================"

python --version

# All benchmark input loading, matrix/model building and solving takes place
# inside this compute-node step. Both methods receive the allocated CPU budget.
srun --ntasks=1 --cpus-per-task="$CPU_BUDGET" --cpu-bind=cores \
    python -u benchmark_line_scopf.py \
    --case "$CASE_NAME" \
    --case-dir "$CASE_DIR" \
    --method both \
    --lp-solver gurobi \
    --qp-solver gurobi \
    --workers auto \
    --qp-threads 1 \
    --cpu-budget "$CPU_BUDGET" \
    --central-threads "$CPU_BUDGET" \
    --thermal-mode "$THERMAL_MODE" \
    --islanding "$ISLANDING" \
    --repeat "$REPEATS" \
    --max-iter 5000 \
    --output-dir "$OUT_DIR"
