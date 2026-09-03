#!/usr/bin/env bash
# Example SLURM job array: one participant per task, then one group-level job.
#
#   sbatch --array=1-$(($(wc -l < participants_list.txt))) scripts/slurm_array.sh config.yaml participants_list.txt
#   sbatch --dependency=afterok:<array job id> scripts/slurm_group.sh config.yaml
#
#SBATCH --job-name=hopfec
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/hopfec_%A_%a.out
set -euo pipefail
CONFIG=${1:?config.yaml}
LIST=${2:?participants_list.txt}   # one participant label per line (sub-01 ...)
SUB=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
source "$(dirname "$0")/../venv/bin/activate"
hopfec timeseries -c "$CONFIG" --participant-label "$SUB" --n-jobs 1
hopfec fit        -c "$CONFIG" --participant-label "$SUB" --participants-only --n-jobs "${SLURM_CPUS_PER_TASK:-8}" --set model.search.level=participant
