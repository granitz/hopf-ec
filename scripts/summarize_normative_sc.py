#!/usr/bin/env python
"""Summarise the Schaefer+Tian normative SC builds: sanity numbers, LAS-vs-RAS mirroring check,
MNI152NLin6Asym vs MNI152NLin2009cAsym agreement.  Usage: python scripts/summarize_normative_sc.py [LAS_dir] [RAS_dir]"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hopfec.atlases import homotopic_key, read_label_table  # noqa: E402
from hopfec.utils import load_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LAS = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "derivatives" / "normative_sc" / "sc"
RAS = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "derivatives" / "normative_sc_RAS" / "sc"
LABELS = {116: ROOT / "data/Tian_atlas/schaefertian100/SchaeferTian116.tsv", 216: ROOT / "data/atlases_tian/Schaefer2018_200Parcels_7Networks_order_Tian_Subcortex_S1_label.txt",
          416: ROOT / "data/Tian_atlas/SchaeferTian416.txt"}


def slug(s: str) -> str:
    import re

    return re.sub(r"[^A-Za-z0-9]", "", s)[:60]


def load(d: Path, name: str, weight: str = "count"):
    f = d / f"atlas-{slug(name)}_desc-normative_weight-{weight}_connectivity.tsv"
    return load_matrix(f) if f.exists() else None


def corr(A, B):
    iu = np.triu_indices(A.shape[0], 1)
    return float(np.corrcoef(np.log1p(A[iu]), np.log1p(B[iu]))[0, 1])


def swap_perm(n: int):
    t = read_label_table(LABELS[n])
    keys: dict = {}
    for i, (nm, h) in enumerate(zip(t["name"], t["hemisphere"])):
        if h in ("L", "R"):
            keys.setdefault(homotopic_key(nm), {})[h] = i
    perm = np.arange(n)
    for d in keys.values():
        if "L" in d and "R" in d:
            perm[d["L"]], perm[d["R"]] = d["R"], d["L"]
    return perm


for n in (116, 216, 416):
    print(f"\n===== SchaeferTian{n} =====")
    for space in ("MNI152NLin6Asym", "MNI152NLin2009cAsym"):
        name = f"SchaeferTian{n}-{space}"
        js = LAS / f"atlas-{slug(name)}_desc-normative_connectivity.json"
        if js.exists():
            s = json.loads(js.read_text())
            print(f"  LAS {space:20s}: streamlines connecting>=2 parcels {s.get('n_connecting_streamlines'):,}  density {s['density']:.3f}  empty nodes {s['empty_nodes']}  "
                  f"L/R row-sum ratio {s.get('LR_ratio', float('nan')):.3f}  interhemispheric {s.get('interhemispheric_fraction', float('nan')):.3f}  max edge {s['max_edge']:.0f}")
    A6 = load(LAS, f"SchaeferTian{n}-MNI152NLin6Asym")
    A9 = load(LAS, f"SchaeferTian{n}-MNI152NLin2009cAsym")
    R9 = load(RAS, f"SchaeferTian{n}-MNI152NLin2009cAsym")
    if A6 is not None and A9 is not None:
        print(f"  6Asym vs 2009cAsym atlas (both LAS): corr(log counts) = {corr(A6, A9):.3f}")
    if A9 is not None and R9 is not None:
        perm = swap_perm(n)
        print(f"  LAS vs RAS (2009cAsym atlas): corr = {corr(A9, R9):.3f};  after swapping L/R homologues in the RAS build: {corr(A9, R9[np.ix_(perm, perm)]):.3f}")
