#!/usr/bin/env bash
#SBATCH --job-name=hopfec-group
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/hopfec_group_%j.out
set -euo pipefail
CONFIG=${1:?config.yaml}
source "$(dirname "$0")/../venv/bin/activate"
# group-level EC from the participant fits produced by the array job (+ group comparisons)
hopfec fit -c "$CONFIG" --group-only --n-jobs "${SLURM_CPUS_PER_TASK:-16}"
