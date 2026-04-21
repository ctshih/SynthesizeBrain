# SynthesizeBrain

Pack warped single-neuron volumes from the FlyCircuit dataset into paired
**intensity + instance-label** volumes, intended as training data for
single-neuron auto-segmentation on dense-staining brain images.

## What it produces

For each chosen `N`, a directory under `output/output_N{requested}_K{achieved}/`
containing (e.g. requesting N=500 and getting K=148 yields `output/output_N500_K148/`):

- `intensity.am` / `intensity.nii.gz` — uint16, voxel values = original warp
  intensities where a selected neuron lives, 0 elsewhere.
- `labels.am` / `labels.nii.gz` — uint16, per-voxel instance label ID (1..K),
  0 = background. Paired 1-to-1 with `intensity.*`.
- `neuron_list.tsv` — `label_id, filename, driver, voxel_count,
  bbox_coverage, origin_{ix,iy,iz}, lattice_{nx,ny,nz}`.
- `mip.png` — three-axis MIP preview (intensity grey; labels random colors).

## Packing constraints

The `N` chosen neurons satisfy:

1. **BBox coverage ≥ 50%.** For each selected neuron *n*,
   `|bbox(n) ∩ ⋃_{m≠n} bbox(m)| / |bbox(n)| ≥ 0.5` (voxels counted once).
   This prevents trivially-separated training cases.
2. **Voxel-level disjointness.** Selected neurons' non-zero voxels are
   pairwise disjoint (touching is fine, overlap is not). Real dense tissue
   doesn't overlap; warp-from-different-fly data can.

Mix of drivers (Tdc2 / Trh / VGlut / fru) is allowed in a single synthesis.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
# 1) One-time scan of the warp directory (cache per-neuron metadata).
python synthesize.py index

# 2) Synthesize one run at a given N.
python synthesize.py synthesize --n 100

# 3) Sweep several N values, emit output/sweep_summary.tsv.
python synthesize.py sweep --ns 10 50 100 500
```

Defaults:

- warp dir: `C:\Users\USER\Work\Kaleido\warp`
- cache:    `cache/warp_index.npz` (plus `cache/candidate_scores.npz` after
            the first selection — this amortises the ~minute-long coverage-
            map + score pass across runs).
- output:   `output/output_N{requested}_K{achieved}/` (name reveals the
            packing ceiling at a glance)
- canvas:   derived from the union of all 9987 per-neuron bboxes (989×646×337
            at 1-voxel spacing for the current dataset).

## Empirical packing ceiling

With the dataset in `Kaleido\warp\` (9987 FlyCircuit warps) and seed=42:

| N requested | K achieved | elapsed | coverage mean / min |
|------------:|-----------:|--------:|---------------------|
| 10          | 10         | ~10 s   | 0.80 / 0.54 |
| 50          | 50         | ~65 s   | 0.90 / 0.51 |
| 100         | 100        | ~65 s   | 0.90 / 0.54 |
| 500         | ~148       | ~80 s   | 0.86 / 0.00 (3 violators) |

The voxel-disjoint packing saturates around **K ≈ 148**: after that the
greedy exhausts all 9987 candidates and the remaining ones all overlap
with something already placed. If you need more, we'd have to relax a
constraint — see `synthesize_brain/select.py` for levers.

## Algorithm sketch

- **Phase 1 — `synthesize_brain/index.py`**: scan every `.am`; record cropped
  lattice dims, header bbox, **tight non-zero bbox**, non-zero voxel count.
  Canvas = union of all header bboxes, rounded to integer voxels.
- **Phase 2 — `synthesize_brain/select.py`**:
  1. Build a canvas-voxel coverage counter across the full dataset; score
     every neuron by mean coverage inside its own tight bbox (central = high).
  2. Greedy accept candidates in decreasing score order (tie-break: smaller
     neurons first), rejecting any whose voxels intersect `occupied_mask`.
     Reads are parallelised across 8 threads with a bounded prefetch
     window; rejected reads are retained in a shared in-memory cache so
     the repair pass does not re-decode them.
  3. Validate constraint (1) using an incrementally-maintained per-voxel
     bbox counter; drop violators, attempt refill from the score-sorted
     pool, iterate up to 5 rounds.
- **Phase 3 — `synthesize_brain/compose.py`**: paste each accepted neuron's
  non-zero voxels into `intensity` and write its label ID into `labels`.
- **Phase 4 — writers**: `amira_io.py` (raw AmiraMesh ushort),
  `nifti_io.py` (.nii.gz via nibabel), `mip.py` (three-axis MIP PNG).

## Repo layout

```
synthesize.py                  # CLI entry point
synthesize_brain/
├── amira_io.py                # AmiraMesh reader (from Kaleido) + ushort writer
├── compose.py                 # Phase 3
├── index.py                   # Phase 1
├── mip.py                     # MIP preview
├── nifti_io.py                # NIfTI writer
└── select.py                  # Phase 2
cache/                         # warp_index.npz, candidate_scores.npz
output/                        # output_N{req}_K{ach}/, sweep_summary.tsv
tasks/                         # todo.md, lessons.md
```
