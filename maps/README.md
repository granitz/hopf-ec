# Parcel-wise brain maps

One value per parcel of an atlas, for `model.heterogeneity` (`a_j = a0 + beta * z_j`):

```yaml
model:
  heterogeneity:
    enabled: true
    map: {name: myelin, file: maps/Schaefer2018_100Parcels_7Networks_order_Tian_Subcortex_S1_myelin.tsv}
    beta: {start: -0.05, stop: 0.05, num: 11}
```

`file` is read relative to the directory `hopfec` is run from (use an absolute path otherwise).
Columns: `index` (atlas label id, matched to the atlas), `name`, and the value column; `n/a` marks parcels
without a value (they are kept at the homogeneous `a0`).

| file | map | parcels with a value |
|---|---|---|
| `Schaefer2018_{100,200,400}Parcels_7Networks_order_Tian_Subcortex_S1_myelin.tsv` | HCP-S1200 group-average T1w/T2w myelin (neuromaps `hcps1200/myelinmap/fsLR/32k`), mean over the vertices of each parcel of the atlas' fsLR-32k dlabel | the cortical Schaefer parcels; the 16 Tian S1 subcortical parcels have no surface value |

Regenerate, or build a table for another atlas / another neuromaps annotation / an MNI-space volume:

```bash
python scripts/make_parcel_map.py --atlas <atlas.nii.gz> --labels <labels.txt> --dlabel <atlas.dlabel.nii> --out maps/<atlas>_myelin.tsv
python scripts/make_parcel_map.py --atlas <atlas.nii.gz> --volume <map_MNI.nii.gz> --name <name> --out maps/<atlas>_<name>.tsv
```

(`pip install neuromaps` for the annotation route.)  Cite Glasser et al. 2016 (Nature) for the myelin map and
Markello et al. 2022 (Nature Methods) for neuromaps.
