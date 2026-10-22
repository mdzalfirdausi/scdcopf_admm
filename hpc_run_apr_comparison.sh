#!/bin/bash
#SBATCH --job-name=apr_compare
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%j_apr_compare.out
#SBATCH --error=%j_apr_compare.err

set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: Submit with sbatch, not bash." >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR:?Missing Slurm submission directory}"
trap 'job_status=$?; echo "Finished: $(date) | Exit code: $job_status"' EXIT
CASE_NAME=""
OUT_DIR=""
EXTRA_ARGS=()
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --with-lp-start|--tee|--no-line-contingencies)
            EXTRA_ARGS+=("$1"); shift ;;
        --case|--output-dir|--case-dir|--gamma|--penalty|--demand-scale|--islanding|--generator-contingencies|--lp-time-limit|--mip-time-limit|--mip-gap|--weight-start|--weight-tol|--weight-rtol|--weight-max|--weight-max-solves|--apr-tol-mw|--feas-tol-mw)
            if [[ "$#" -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "ERROR: $1 requires a value." >&2; exit 1
            fi
            case "$1" in
                --case) CASE_NAME="$2" ;;
                --output-dir) OUT_DIR="$2" ;;
                *) EXTRA_ARGS+=("$1" "$2") ;;
            esac
            shift 2 ;;
        *) echo "ERROR: Unknown argument: $1" >&2; exit 1 ;;
    esac
done
if [[ -z "$CASE_NAME" ]]; then
    echo "Usage: sbatch hpc_run_apr_comparison.sh --case CASE [--output-dir PATH] [--with-lp-start] [--mip-time-limit SECONDS]" >&2
    exit 1
fi
if [[ ! -f compare_apr_lp_mip.py ]]; then
    echo "ERROR: compare_apr_lp_mip.py is missing from $PWD." >&2; exit 1
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
CASE_TAG="${CASE_NAME##*/}"
CASE_TAG="${CASE_TAG%.xlsx}"
if [[ -z "$OUT_DIR" ]]; then
    OUT_DIR="$PWD/output_apr_compare_${CASE_TAG}_${SLURM_JOB_ID}"
fi
mkdir -p "$OUT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Compute node: $(hostname)"
echo "CPUs: $CPU_BUDGET"
echo "Case: $CASE_NAME"
echo "Output: $OUT_DIR"
echo "Python: $(command -v python)"
echo "Start time: $(date)"
python --version
srun --ntasks=1 --cpus-per-task="$CPU_BUDGET" --cpu-bind=cores \
    python -u compare_apr_lp_mip.py \
    --case "$CASE_NAME" \
    --solver gurobi \
    --threads "$CPU_BUDGET" \
    --output-dir "$OUT_DIR" \
    "${EXTRA_ARGS[@]}"
