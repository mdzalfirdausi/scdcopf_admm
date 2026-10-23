#!/bin/bash
#SBATCH --job-name=apr_sigma
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4%3
#SBATCH --output=%A_%a_apr_sigma.out
#SBATCH --error=%A_%a_apr_sigma.err

set -euo pipefail
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || -z "${SLURM_ARRAY_JOB_ID:-}" ]]; then
    echo "ERROR: Submit this script as a Slurm array with sbatch." >&2
    exit 1
fi
cd "${SLURM_SUBMIT_DIR:?Missing submission directory}"
CASES=(
    pglib_opf_case200_activ
    pglib_opf_case240_pserc
    pglib_opf_case300_ieee
    pglib_opf_case500_goc
    pglib_opf_case588_sdet
)
if [[ ! "$SLURM_ARRAY_TASK_ID" =~ ^[0-4]$ ]]; then
    echo "ERROR: Array task index must be 0 through 4." >&2
    exit 1
fi
if [[ ! -f hpc_run_apr_sigma.sh ]]; then
    echo "ERROR: hpc_run_apr_sigma.sh is missing from $PWD." >&2
    exit 1
fi
CASE_NAME="${CASES[$SLURM_ARRAY_TASK_ID]}"
OUT_DIR="$PWD/output_table_B_${CASE_NAME}_hpc_array_${SLURM_ARRAY_JOB_ID}"
echo "Array: $SLURM_ARRAY_JOB_ID; task: $SLURM_ARRAY_TASK_ID; case: $CASE_NAME"
echo "Output: $OUT_DIR"
# Reuse the working KFUPM environment setup and srun launch. Its #SBATCH lines
# are comments when invoked through bash; this array's allocation applies.
bash hpc_run_apr_sigma.sh \
    --uncertainty-dir uncertainty \
    --seed 19 --samples 100 \
    --penalty-per-mw 1500 --weight-start 150000 \
    --lp-time-limit 300 --relaxation-time-limit 600 \
    "$@" \
    --case "${CASE_NAME}.xlsx" --output-dir "$OUT_DIR"
