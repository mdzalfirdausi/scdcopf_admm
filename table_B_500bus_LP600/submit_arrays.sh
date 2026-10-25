#!/bin/bash
set -euo pipefail
sbatch --array=0-49%32 --cpus-per-task=16 --mem=64G --time=24:00:00 --output=/home/g202210120/projects/scdcopf_admm/table_B_500bus_LP600/logs/small_%A_%a.out --error=/home/g202210120/projects/scdcopf_admm/table_B_500bus_LP600/logs/small_%A_%a.err /home/g202210120/projects/scdcopf_admm/table_B_500bus_LP600/code/hpc_run_apr_sigma_batches.sh /home/g202210120/projects/scdcopf_admm/table_B_500bus_LP600 small
