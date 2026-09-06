#!/usr/bin/env python
"""Write a per-parcel brain map (default: HCP-S1200 T1w/T2w myelin from neuromaps) as a TSV for
``model.heterogeneity.map: {name: myelin, file: maps/<atlas>_myelin.tsv}``.

Surface annotations are parcellated with the atlas' CIFTI dlabel (fsLR 32k); parcels that are not on
the surface (Tian subcortex) get NaN and stay at the homogeneous a0 in the fit.

    python scripts/make_parcel_map.py \\
        --atlas data/atlases_tian/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_3T_MNI152NLin2009cAsym_2mm.nii.gz \\
        --labels data/atlases_tian/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_label.txt \\
        --dlabel data/Tian_atlas/schaefertian100/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1.dlabel.nii \\
        --out maps/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_myelin.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hopfec.atlases import atlas_from_file
from hopfec.maps import load_parcel_map


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atlas", required=True, help="atlas volume (NIfTI) that the fit uses")
    ap.add_argument("--labels", default=None, help="label table of the atlas (if not auto-detected)")
    ap.add_argument("--dlabel", default=None, help="CIFTI dlabel of the same atlas (needed for surface annotations)")
    ap.add_argument("--volume", default=None, help="instead of neuromaps: an MNI-space NIfTI map, parcellated with --atlas")
    ap.add_argument("--name", default="myelin")
    ap.add_argument("--source", default="hcps1200")
    ap.add_argument("--desc", default="myelinmap")
    ap.add_argument("--space", default="fsLR")
    ap.add_argument("--den", default="32k")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    atlas = atlas_from_file(args.atlas, labels=args.labels)
    if args.volume:
        spec = {"name": args.name, "file": args.volume}
    else:
        spec = {"name": args.name, "neuromaps": {"source": args.source, "desc": args.desc, "space": args.space, "den": args.den},
                "surface_labels": args.dlabel}
    pm = load_parcel_map(spec, atlas)
    df = pd.DataFrame({"index": [int(i) for i in atlas.label_ids], "name": list(atlas.region_names), args.name: pm.values})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False, float_format="%.6g", na_rep="n/a")
    ok = np.isfinite(pm.values)
    print(f"{out}: {ok.sum()}/{len(ok)} parcels with a value from {pm.source}"
          + (f" ({(~ok).sum()} without: {', '.join(np.asarray(atlas.region_names)[~ok][:6])}{'...' if (~ok).sum() > 6 else ''})" if (~ok).any() else ""))


if __name__ == "__main__":
    main()
