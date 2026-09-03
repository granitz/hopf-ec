"""Catalogue of parcellation atlases in a folder.

An atlas is a 3-D integer label NIfTI plus (optionally) a label table.  Supported label tables:

* BIDS/TemplateFlow ``*_dseg.tsv`` (columns ``index``, ``name`` [, ``hemisphere``, ``network`` ...])
* generic TSV/CSV with an id column (index/id/label/value) and a name column
* FreeSurfer-style LUT ``.txt`` (``id name R G B A``)
* plain text, one region name per line (ids are then 1..N)

The template space is taken from the ``space-``/``tpl-`` filename entity when present, or
inferred from the voxel grid (bounding box in world mm) for the common MNI grids.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import nibabel as nib
import numpy as np
import pandas as pd

from .bids import parse_entities, strip_ext
from .utils import LOG

# world-space bounding boxes (mm) of well-known MNI grids: (space, shape, voxel size, min corner, max corner)
KNOWN_GRIDS = [
    ("MNI152NLin6Asym", (91, 109, 91), 2.0, (-90.0, -126.0, -72.0), (90.0, 90.0, 108.0)),
    ("MNI152NLin6Asym", (182, 218, 182), 1.0, (-91.0, -126.0, -72.0), (90.0, 91.0, 109.0)),
    ("MNI152NLin2009cAsym", (97, 115, 97), 2.0, (-96.5, -132.5, -78.5), (95.5, 95.5, 113.5)),
    ("MNI152NLin2009cAsym", (193, 229, 193), 1.0, (-96.0, -132.0, -78.0), (96.0, 96.0, 114.0)),
]

HEMI_TOKENS = {
    "L": ["LH", "L", "Left", "left", "lh"],
    "R": ["RH", "R", "Right", "right", "rh"],
}
_HEMI_RE = re.compile(r"(^|[_\-\s.])(LH|RH|lh|rh|L|R|Left|Right|left|right)(?=$|[_\-\s.])")


def infer_hemisphere(name: str) -> str:
    """Return 'L', 'R' or '' from a region name such as '7Networks_LH_Vis_1' or 'HIP-rh'."""
    m = _HEMI_RE.search(str(name))
    if not m:
        return ""
    tok = m.group(2)
    return "L" if tok in HEMI_TOKENS["L"] else "R"


def homotopic_key(name: str) -> str:
    return _HEMI_RE.sub(lambda m: m.group(1) + "*", str(name))


@dataclass
class Atlas:
    name: str
    image: Path
    labels_file: Path | None = None
    space: str | None = None
    space_source: str = "unknown"
    resolution_mm: tuple[float, float, float] | None = None
    shape: tuple[int, int, int] | None = None
    label_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    region_names: list[str] = field(default_factory=list)
    hemispheres: list[str] = field(default_factory=list)
    networks: list[str] = field(default_factory=list)
    label_table: pd.DataFrame | None = None

    @property
    def n_parcels(self) -> int:
        return int(len(self.label_ids))

    def load_img(self) -> nib.Nifti1Image:
        return nib.load(str(self.image))

    def data(self) -> np.ndarray:
        d = np.asanyarray(self.load_img().dataobj)
        d = np.rint(d).astype(np.int32)
        d[d < 0] = 0
        return d

    def homotopic_pairs(self) -> list[tuple[int, int]]:
        """Index pairs (i, j) of left/right homologous regions (0-based positions in label order)."""
        if not self.region_names or not any(self.hemispheres):
            return []
        keys: dict[str, dict[str, int]] = {}
        for i, (nm, h) in enumerate(zip(self.region_names, self.hemispheres)):
            if h not in ("L", "R"):
                continue
            keys.setdefault(homotopic_key(nm), {})[h] = i
        pairs = [(d["L"], d["R"]) for d in keys.values() if "L" in d and "R" in d]
        return sorted(pairs)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "image": str(self.image),
            "labels_file": str(self.labels_file) if self.labels_file else None,
            "space": self.space,
            "space_source": self.space_source,
            "resolution_mm": list(self.resolution_mm) if self.resolution_mm else None,
            "shape": list(self.shape) if self.shape else None,
            "n_parcels": self.n_parcels,
            "n_left": int(sum(h == "L" for h in self.hemispheres)),
            "n_right": int(sum(h == "R" for h in self.hemispheres)),
            "n_homotopic_pairs": len(self.homotopic_pairs()),
            "networks": sorted(set(n for n in self.networks if n)),
        }


# --------------------------------------------------------------------------- label tables
def _pick_col(cols: Sequence[str], candidates: Sequence[str]) -> str | None:
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low:
            return low[cand]
    return None


def read_label_table(path: str | Path) -> pd.DataFrame:
    """Return DataFrame with columns index, name [, hemisphere, network, type ...]."""
    path = Path(path)
    text = path.read_text().strip("\n")
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    if not lines:
        raise ValueError(f"empty label file {path}")
    sep = "\t" if "\t" in lines[0] else ("," if path.suffix.lower() == ".csv" or "," in lines[0] else None)
    df = None
    first_tok = lines[0].split(sep)[0].strip() if sep else lines[0].split()[0]
    headerless = re.fullmatch(r"-?\d+", first_tok) is not None and sep is None
    try:
        if headerless:
            raise ValueError("no header")
        df = pd.read_csv(path, sep=sep, engine="python", comment="#", dtype=str)
        if df.shape[1] < 2:
            df = None
    except Exception:  # noqa: BLE001
        df = None
    if df is not None:
        idc = _pick_col(df.columns, ["index", "id", "value", "roi", "region_id", "labelid", "label_id", "label"])
        namec = _pick_col(df.columns, ["name", "region", "roi_name", "region_name", "label_name", "labelname", "label"])
        if idc is None or namec is None:
            # unnamed / numeric headers (e.g. a pandas dump): guess an integer id column and a text column
            df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
            def _is_int(col):
                return pd.to_numeric(col, errors="coerce").notna().all() and (pd.to_numeric(col, errors="coerce") % 1 == 0).all()
            ints = [c for c in df.columns if _is_int(df[c])]
            texts = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").isna().mean() > 0.5]
            uniq = [c for c in ints if df[c].nunique() == len(df) and pd.to_numeric(df[c]).min() >= 1]
            if idc is None and uniq and texts:
                idc = uniq[0]
            if namec is None and texts:
                namec = texts[0]
            if idc is not None and namec is not None and str(idc).strip() == "":
                idc = uniq[1] if len(uniq) > 1 else idc
            if idc is None or namec is None or idc == namec:
                df = None
    if df is not None:
        if namec == idc:
            others = [c for c in df.columns if c != idc]
            namec = others[0] if others else None
        out = pd.DataFrame({"index": pd.to_numeric(df[idc], errors="coerce")})
        out["name"] = df[namec].astype(str) if namec else [f"roi_{i}" for i in out["index"]]
        hc = _pick_col(df.columns, ["hemisphere", "hemi", "hem", "side"])
        out["hemisphere"] = df[hc].astype(str).str.strip().str.upper().str[0].where(df[hc].notna(), "") if hc else [infer_hemisphere(n) for n in out["name"]]
        out["hemisphere"] = out["hemisphere"].map(lambda h: "L" if h in ("L", "l") else "R" if h in ("R", "r") else "")
        nc = _pick_col(df.columns, ["network", "yeo_network", "net", "system"])
        out["network"] = df[nc].astype(str) if nc else ""
        tc = _pick_col(df.columns, ["type", "structure", "tissue"])
        if tc:
            out["type"] = df[tc].astype(str)
        out = out.dropna(subset=["index"])
        out["index"] = out["index"].astype(int)
        return out.reset_index(drop=True)
    # Connectome Workbench label list: name on one line, "key R G B A" on the next
    if len(lines) >= 2 and len(lines) % 2 == 0 and all(re.fullmatch(r"\s*\d+(\s+\d+){3,4}\s*", lines[i]) for i in range(1, len(lines), 2)) \
            and not any(re.fullmatch(r"\s*\d+(\s+\S+)+\s*", lines[i]) and re.fullmatch(r"-?\d+", lines[i].split()[0]) for i in range(0, len(lines), 2)):
        rows = [(int(lines[i + 1].split()[0]), lines[i].strip()) for i in range(0, len(lines), 2)]
        out = pd.DataFrame(rows, columns=["index", "name"])
        out["hemisphere"] = [infer_hemisphere(n) for n in out["name"]]
        out["network"] = ""
        return out.sort_values("index").reset_index(drop=True)
    # FreeSurfer LUT or plain list
    rows = []
    for k, ln in enumerate(lines, start=1):
        toks = ln.split()
        if len(toks) >= 2 and re.fullmatch(r"-?\d+", toks[0]):
            rows.append((int(toks[0]), toks[1]))
        else:
            rows.append((k, ln.strip()))
    out = pd.DataFrame(rows, columns=["index", "name"])
    out["hemisphere"] = [infer_hemisphere(n) for n in out["name"]]
    out["network"] = ""
    return out


def find_label_file(image: Path) -> Path | None:
    stem, _ = strip_ext(image.name)
    d = image.parent
    cands = [d / f"{stem}{ext}" for ext in (".tsv", ".txt", ".csv")]
    cands += [d / f"{stem}_labels{ext}" for ext in (".tsv", ".txt", ".csv")]
    stem_nores = re.sub(r"_res-[A-Za-z0-9]+", "", stem)
    if stem_nores != stem:
        cands += [d / f"{stem_nores}{ext}" for ext in (".tsv", ".txt", ".csv")]
    stem_nospace = re.sub(r"_(space|tpl)-[A-Za-z0-9]+", "", stem_nores)
    if stem_nospace != stem_nores:
        cands += [d / f"{stem_nospace}{ext}" for ext in (".tsv", ".txt", ".csv")]
    for c in cands:
        if c.exists():
            return c
    # same atlas/desc/seg entities in another template space (label names are space independent)
    ents = parse_entities(image.name)
    keys = [k for k in ("atlas", "desc", "seg") if k in ents]
    if keys:
        for c in sorted(d.glob("*.tsv")) + sorted(d.glob("*.txt")) + sorted(d.glob("*.csv")):
            ce = parse_entities(c.name)
            if all(ce.get(k) == ents[k] for k in keys) and ce.get("suffix") in ("dseg", "labels", None, ents.get("suffix")):
                return c
    # loose match: a *labels* table whose name shares the atlas core name
    core = re.sub(r"_(space|tpl|res)-[A-Za-z0-9]+", "", stem)
    core = re.sub(r"_dseg$", "", core)
    for c in sorted(d.glob("*.tsv")) + sorted(d.glob("*.txt")) + sorted(d.glob("*.csv")):
        cstem, _ = strip_ext(c.name)
        cstem = re.sub(r"_(labels|dseg|lut)$", "", cstem, flags=re.I)
        if cstem and (cstem in core or core in cstem):
            return c
    return None


# --------------------------------------------------------------------------- space inference
def infer_space_from_grid(img: nib.Nifti1Image) -> str | None:
    shape = tuple(int(s) for s in img.shape[:3])
    aff = img.affine
    corners = np.array([[0, 0, 0, 1], [shape[0] - 1, shape[1] - 1, shape[2] - 1, 1]], float)
    world = (aff @ corners.T).T[:, :3]
    lo, hi = world.min(axis=0), world.max(axis=0)
    for space, kshape, _vs, kmin, kmax in KNOWN_GRIDS:
        if shape == kshape and np.allclose(lo, kmin, atol=0.01) and np.allclose(hi, kmax, atol=0.01):
            return space
    return None


def atlas_from_file(image: str | Path, labels: str | Path | None = None, name: str | None = None) -> Atlas:
    image = Path(image)
    img = nib.load(str(image))
    ents = parse_entities(image.name)
    stem, _ = strip_ext(image.name)
    space = ents.get("space") or ents.get("tpl")
    src = "filename" if space else "unknown"
    if name is None:
        parts = [f"{k}-{v}" for k, v in ents.items() if k in ("atlas", "desc", "seg")]
        if parts and space:
            parts.append(f"space-{space}")
        name = "_".join(parts) if parts else stem
    if space is None:
        space = infer_space_from_grid(img)
        src = "grid" if space else "unknown"
    vs = tuple(float(x) for x in img.header.get_zooms()[:3])
    data = np.asanyarray(img.dataobj)
    ids = np.unique(np.rint(data).astype(np.int64))
    ids = ids[ids > 0]
    lab_path = Path(labels) if labels else find_label_file(image)
    names, hemis, nets, table = [], [], [], None
    if lab_path is not None:
        try:
            table = read_label_table(lab_path)
            lut = dict(zip(table["index"], table["name"]))
            hl = dict(zip(table["index"], table["hemisphere"]))
            nl = dict(zip(table["index"], table["network"]))
            missing = [int(i) for i in ids if int(i) not in lut]
            if missing:
                LOG.warning("%s: %d labels in the image have no entry in %s (e.g. %s)", image.name, len(missing), lab_path.name, missing[:5])
            names = [str(lut.get(int(i), f"roi_{int(i)}")) for i in ids]
            hemis = [str(hl.get(int(i), infer_hemisphere(lut.get(int(i), "")))) for i in ids]
            nets = [str(nl.get(int(i), "")) for i in ids]
        except Exception as e:  # noqa: BLE001
            LOG.warning("could not parse label table %s: %s", lab_path, e)
            lab_path = None
    if not names:
        names = [f"roi_{int(i)}" for i in ids]
        hemis = ["" for _ in ids]
        nets = ["" for _ in ids]
    return Atlas(name=name, image=image, labels_file=lab_path, space=space, space_source=src,
                 resolution_mm=vs, shape=tuple(int(s) for s in img.shape[:3]), label_ids=ids,
                 region_names=names, hemispheres=hemis, networks=nets, label_table=table)


def scan_atlas_dir(atlas_dir: str | Path, pattern: str | None = None) -> list[Atlas]:
    """List all label images (``*.nii``, ``*.nii.gz``) in a folder (recursively)."""
    atlas_dir = Path(atlas_dir)
    if not atlas_dir.is_dir():
        raise FileNotFoundError(f"atlas folder not found: {atlas_dir}")
    files = sorted(set(list(atlas_dir.rglob("*.nii.gz")) + list(atlas_dir.rglob("*.nii"))))
    out = []
    for f in files:
        if pattern and not re.search(pattern, f.name):
            continue
        try:
            img = nib.load(str(f))
            if len(img.shape) not in (3, 4) or (len(img.shape) == 4 and img.shape[3] != 1):
                LOG.debug("skipping non-3D image %s", f.name)
                continue
            if "probseg" in f.name or "T1w" in f.name or "mask" in f.name.lower():
                continue
            out.append(atlas_from_file(f))
        except Exception as e:  # noqa: BLE001
            LOG.warning("skipping %s: %s", f, e)
    return out


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _prefer_space(hits: list[Atlas], prefer_space: str | None) -> list[Atlas]:
    if prefer_space and len(hits) > 1:
        same = [a for a in hits if a.space and _norm(a.space) == _norm(prefer_space)]
        if same:
            return same
    return hits


def select_atlas(atlas_dir: str | Path | None, name: str | None = None, file: str | Path | None = None,
                 labels: str | Path | None = None, prefer_space: str | None = None) -> Atlas:
    """Pick one atlas: explicit file, or by name (exact, regex or normalised substring) within atlas_dir."""
    if file:
        return atlas_from_file(file, labels=labels, name=name)
    if atlas_dir is None:
        raise ValueError("either atlas.file or paths.atlas_dir must be set")
    atlases = scan_atlas_dir(atlas_dir)
    if not atlases:
        raise FileNotFoundError(f"no atlases found in {atlas_dir}")
    if name is None:
        if len(atlases) == 1:
            return atlases[0]
        raise ValueError("several atlases available; set atlas.name: " + ", ".join(a.name for a in atlases))
    exact = [a for a in atlases if a.name == name or a.image.name == name or strip_ext(a.image.name)[0] == name]
    exact = _prefer_space(exact, prefer_space)
    if len(exact) == 1:
        return exact[0]
    try:
        rx = re.compile(name)
        hits = [a for a in atlases if rx.search(a.name) or rx.search(a.image.name)]
    except re.error:
        hits = []
    if not hits:
        toks = [t for t in re.split(r"[^a-z0-9]+", str(name).lower()) if t]
        hits = [a for a in atlases if toks and (all(t in _norm(a.name) for t in toks) or all(t in _norm(a.image.name) for t in toks))]
    hits = _prefer_space(hits, prefer_space)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise ValueError(f"no atlas matching {name!r} in {atlas_dir}; available: " + ", ".join(a.name for a in atlases))
    raise ValueError(f"atlas name {name!r} is ambiguous (set atlas.file or a more specific name): " + ", ".join(a.image.name for a in hits))


def atlases_table(atlases: Iterable[Atlas]) -> pd.DataFrame:
    rows = []
    for a in atlases:
        s = a.summary()
        rows.append({
            "name": s["name"],
            "space": s["space"] or "?",
            "res_mm": "x".join(f"{v:g}" for v in (s["resolution_mm"] or [])),
            "shape": "x".join(str(v) for v in (s["shape"] or [])),
            "n_parcels": s["n_parcels"],
            "L/R": f"{s['n_left']}/{s['n_right']}",
            "homotopic_pairs": s["n_homotopic_pairs"],
            "labels": Path(s["labels_file"]).name if s["labels_file"] else "-",
            "file": Path(s["image"]).name,
        })
    return pd.DataFrame(rows)


def write_labels_tsv(atlas: Atlas, path: str | Path) -> Path:
    df = pd.DataFrame({"index": atlas.label_ids, "name": atlas.region_names,
                       "hemisphere": atlas.hemispheres, "network": atlas.networks})
    df.to_csv(path, sep="\t", index=False)
    return Path(path)
