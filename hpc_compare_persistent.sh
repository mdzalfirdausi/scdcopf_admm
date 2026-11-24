#!/bin/bash
#SBATCH --job-name=apr_persistent
#SBATCH --partition=cpu_x440
#SBATCH --exclude=node0032
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%j_apr_persistent.out
#SBATCH --error=%j_apr_persistent.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the project directory}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
python -c 'import highspy, gurobipy; print("HiGHS",highspy.Highs().version()); print("Gurobi",gurobipy.gurobi.version())'
echo "Job: $SLURM_JOB_ID; node: $(hostname); start: $(date)"
lscpu
for solver in highs; do
    output_dir="$PWD/output_persistent_${solver}_300_${SLURM_JOB_ID}"
    srun --ntasks=1 --cpus-per-task=1 --cpu-bind=cores \
        python -u compare_apr_lp_mip_persistent.py \
        --lp-only --case pglib_opf_case300_ieee.xlsx \
        --case-dir ../excel_outputs --solver "$solver" --threads 1 \
        --gamma 1 --penalty 150000 --weight-start 150000 \
        --weight-tol 0.001 --weight-rtol 0 --weight-max 1000000 \
        --weight-max-solves 40 --lp-time-limit 300 \
        --apr-tol-mw 0.0001 --feas-tol-mw 0.0001 \
        --output-dir "$output_dir"
done
echo "Finished: $(date)"
