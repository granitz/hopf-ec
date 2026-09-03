#!/usr/bin/env python
"""Compare two ROI x ROI structural connectivity matrices (e.g. hopfec output vs an older MATLAB result).

    python scripts/compare_sc.py A.tsv B.mat [--key conn_matrix] [--labels SchaeferTian416.txt]

Reports edge-wise correlation, and whether B matches A better after swapping left/right homologues
(detects a left-right mirrored connectome when a label table with hemispheres is given).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hopfec.atlases import read_label_table, homotopic_key  # noqa: E402
from hopfec.utils import load_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--key", help="variable name inside a .mat file (default: conn_matrix / first square matrix)")
    ap.add_argument("--key-b", help="variable name inside the second .mat file")
    ap.add_argument("--labels", help="label table (index/name or one name per line) to test a left-right swap")
    args = ap.parse_args()
    A = load_matrix(args.a, args.key or ("conn_matrix" if args.a.endswith(".mat") else None))
    B = load_matrix(args.b, args.key_b or args.key or ("conn_matrix" if args.b.endswith(".mat") else None))
    assert A.shape == B.shape, f"shapes differ: {A.shape} vs {B.shape}"
    n = A.shape[0]
    iu = np.triu_indices(n, 1)

    def corr(X, Y):
        x, y = np.log1p(X[iu]), np.log1p(Y[iu])
        return float(np.corrcoef(x, y)[0, 1])

    print(f"n = {n}; density A {np.count_nonzero(A[iu]) / len(iu[0]):.3f}, B {np.count_nonzero(B[iu]) / len(iu[0]):.3f}")
    print(f"corr(log1p A, log1p B) = {corr(A, B):.4f};  identical: {np.allclose(A, B)}")
    if args.labels:
        t = read_label_table(args.labels)
        names, hemi = list(t["name"]), list(t["hemisphere"])
        keys: dict = {}
        for i, (nm, h) in enumerate(zip(names, hemi)):
            if h in ("L", "R"):
                keys.setdefault(homotopic_key(nm), {})[h] = i
        perm = np.arange(n)
        for d in keys.values():
            if "L" in d and "R" in d:
                perm[d["L"]], perm[d["R"]] = d["R"], d["L"]
        Bs = B[np.ix_(perm, perm)]
        print(f"corr after swapping L/R homologues in B = {corr(A, Bs):.4f}  ({int((perm != np.arange(n)).sum())} regions swapped)")
        if corr(A, Bs) > corr(A, B) + 0.02:
            print("=> B looks LEFT-RIGHT MIRRORED relative to A")


if __name__ == "__main__":
    main()
