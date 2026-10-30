#!/bin/bash
#SBATCH --job-name=apr_warm
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032,node0040
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=%j_apr_warm.out
#SBATCH --error=%j_apr_warm.err
set -eo pipefail
if [[ -z "${SLURM_JOB_ID:-}" || "$#" -ne 3 ]]; then
    echo 'Usage: sbatch hpc_run_apr_warm_only.sh CASE.xlsx SAVED_LP_DIRECTORY EXPECTED_OBJECTIVE' >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
module purge
module load gurobi/13.0.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1
CASE_NAME="$1"
LP_SOURCE="$2"
EXPECTED_OBJECTIVE="$3"
CASE_TAG="${CASE_NAME##*/}"
CASE_TAG="${CASE_TAG%.xlsx}"
OUT_DIR="$PWD/output_apr_warm_only_${CASE_TAG}_${SLURM_JOB_ID}"
srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --cpu-bind=cores \
  /usr/bin/time -v python -u run_apr_warm_only.py \
  --warm-only --lp-source "$LP_SOURCE" \
  --expected-start-objective "$EXPECTED_OBJECTIVE" \
  --case "$CASE_NAME" --case-dir ../excel_outputs \
  --solver gurobi --threads "$SLURM_CPUS_PER_TASK" \
  --gamma 1 --penalty 150000 --islanding skip \
  --mip-time-limit 1800 --mip-gap 1e-7 \
  --apr-tol-mw 1e-4 --feas-tol-mw 1e-4 \
  --output-dir "$OUT_DIR" --tee
