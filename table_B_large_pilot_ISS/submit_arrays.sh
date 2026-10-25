#!/bin/bash
set -euo pipefail
sbatch --array=0-2%1 --cpus-per-task=16 --mem=128G --time=24:00:00 --output=/nfs/mfirdausi/project/scdcopf_admm/table_B_large_pilot_ISS/logs/large_%A_%a.out --error=/nfs/mfirdausi/project/scdcopf_admm/table_B_large_pilot_ISS/logs/large_%A_%a.err /nfs/mfirdausi/project/scdcopf_admm/table_B_large_pilot_ISS/code/hpc_run_apr_sigma_batches.sh /nfs/mfirdausi/project/scdcopf_admm/table_B_large_pilot_ISS large
