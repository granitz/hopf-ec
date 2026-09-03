#!/usr/bin/env bash
# Build dTOR-985 normative SC for the Schaefer+Tian S1 atlases (116, 216, 416 parcels) in both
# MNI152NLin6Asym (native grid of the fibers, recommended) and MNI152NLin2009cAsym (affine-aligned,
# as in the original scripts).  Usage: scripts/build_normative_sc_tian.sh <RAS|LAS|auto> [work_dir] [out_dir]
set -euo pipefail
GRID=${1:-auto}
WORK=${2:-/Volumes/Transcend/hopfec_work}
OUT=${3:-$HOME/hopf-ec/derivatives/normative_sc}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT/venv/bin/activate"
CONN="$ROOT/data/dTOR_fibers_vox_2_mm.mat.gz"
for n in 100 200 400; do
  N=$((n + 16))
  case "$n" in 400) LAB="$ROOT/data/Tian_atlas/SchaeferTian416.txt";; 100) LAB="$ROOT/data/Tian_atlas/schaefertian100/SchaeferTian116.tsv";; *) LAB="$ROOT/data/atlases_tian/Schaefer2018_${n}Parcels_7Networks_order_Tian_Subcortex_S1_label.txt";; esac
  for SP in MNI152NLin6Asym 3T_MNI152NLin2009cAsym; do
    FILE="$ROOT/data/atlases_tian/Schaefer2018_${n}Parcels_7Networks_order_Tian_Subcortex_S1_${SP}_2mm.nii.gz"
    NAME="SchaeferTian${N}-${SP#3T_}"
    echo "=== $NAME ($GRID) ==="
    hopfec sc normative --connectome "$CONN" --fiber-grid "$GRID" -q \
      --set paths.root="$ROOT" --set paths.work_dir="$WORK" --set paths.output_dir="$OUT" \
      --set atlas.file="$FILE" --set atlas.labels="$LAB" --set atlas.name="$NAME" \
      --set sc.normative.chunk_size=200000 --set sc.normative.warp_atlas=never
  done
done
echo "outputs in $OUT/sc"
