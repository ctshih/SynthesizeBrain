# SynthesizeBrain

Pack warped single-neuron volumes from the FlyCircuit dataset into paired
**intensity + instance-label** volumes, intended as training data for
single-neuron auto-segmentation on dense-staining brain images.

## What it produces

For each run, a directory named
`output/output_N{req}_K{c1}_R{total}_s{seed}/` containing:

- `N` = neurons requested on the command line
- `K` = accepted under C1 (greedy + repair survivors)
- `R` = total rendered = K + expand-phase additions (R ≥ K; the expand
  pass admits voxel-disjoint candidates even if they violate C1, for
  denser training volumes)
- `s` = resolved random seed (pass `--seed s` to reproduce)

- `intensity.am` / `intensity.nii.gz` — uint16, voxel values = original warp
  intensities where a selected neuron lives, 0 elsewhere.
- `labels.am` / `labels.nii.gz` — uint16, per-voxel instance label ID (1..K),
  0 = background. Paired 1-to-1 with `intensity.*`.
- `neuron_list.tsv` — `label_id, filename, driver, phase, voxel_count,
  bbox_coverage, origin_{ix,iy,iz}, lattice_{nx,ny,nz}`. The `phase` column
  is `greedy` / `repair` / `expand` — see "Selection phases" below.
- `contacts.csv` — pairwise F/E/V contact voxel counts between every pair of
  neurons that touch at least once. Columns:
  `neuron1, neuron2, N_F, N_E, N_V`.
  - **F** = face-adjacent (share a full face of their unit voxel cubes)
  - **E** = edge-adjacent only (share an edge but not a face)
  - **V** = vertex-adjacent only (share a corner vertex but not an edge)
  Non-touching pairs are omitted.
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

# 2) Synthesize one run (defaults: N=500, random seed each time).
python synthesize.py synthesize

# 3) Reproduce a specific run by passing its seed.
python synthesize.py synthesize --seed 2454876261

# 4) Generate N training datasets with different neuron sets (omit --seed).
for i in 1 2 3 4 5; do python synthesize.py synthesize; done

# 5) Recompute contacts.csv on an existing output dir.
python synthesize.py contacts --dir output/output_N500_K148/

# 6) Sweep several N values.
python synthesize.py sweep --ns 10 50 100 500
```

Defaults:

- warp dir: `C:\Users\USER\Work\Kaleido\warp`
- cache:    `cache/warp_index.npz` (plus `cache/candidate_scores.npz` after
            the first selection — this amortises the ~minute-long coverage-
            map + score pass across runs).
- output:   `output/output_N{req}_K{c1}_R{total}_s{seed}/` (reveals the
            C1-satisfying packing size K, the total-rendered size R after
            the expand pass, and lets randomized runs coexist)
- canvas:   derived from the union of all 9987 per-neuron bboxes (989×646×337
            at 1-voxel spacing for the current dataset).

## Selection phases

The selector works in three phases, recorded per-neuron in
`neuron_list.tsv → phase`:

1. **greedy** — decreasing-score pack until either N is reached or the
   sorted candidate pool is exhausted. Respects C1 + C2 in the sense that
   voxel-disjointness is enforced, but C1 is *not* checked yet.
2. **repair** — validate C1 per-neuron; drop violators (free their voxels
   and bbox contribution), refill their slots from the remaining pool.
   Loops up to 5 rounds. Only neurons that satisfy C1 survive here.
3. **expand** — once no more neurons can replace violators, make one final
   sweep through what's left and admit any that still fit by voxel
   disjointness alone, *ignoring* C1. These are bonus instances for
   training richness; they do not contribute to C1 guarantees.

Set `--rand-sigma 0` to disable the score noise and recover the purely
deterministic selection order.

## Empirical packing ceiling

With the dataset in `Kaleido\warp\` (9987 FlyCircuit warps):

| N requested | K (under C1) | R (total) | notes |
|------------:|-------------:|----------:|-------|
| 10          | 10           | 10        | trivially satisfies C1 |
| 50          | 50           | 50        | C1 OK |
| 100         | 100          | 100       | C1 OK |
| 500         | ~130–150     | ~135–160  | varies per seed; voxel packing saturates |

Voxel-disjoint packing saturates around **R ≈ 140–160** — after that no
remaining neuron fits without overlap. A random-seed run explores a
different local optimum each time, so repeated runs at N=500 give
different K/R values and different neuron subsets, which is the
intended behaviour for generating many training datasets.

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
  `nifti_io.py` (.nii.gz via nibabel), `mip.py` (three-axis MIP PNG),
  `contacts.py` (pairwise F/E/V voxel-touch counts).

## Repo layout

```
synthesize.py                  # CLI entry point
synthesize_brain/
├── amira_io.py                # AmiraMesh reader (from Kaleido) + ushort writer
├── compose.py                 # Phase 3
├── contacts.py                # Pairwise F/E/V contact counts
├── index.py                   # Phase 1
├── mip.py                     # MIP preview
├── nifti_io.py                # NIfTI writer
└── select.py                  # Phase 2
check_saturation.py            # Diagnostic: is final K truly voxel-saturated?
cache/                         # warp_index.npz, candidate_scores.npz
output/                        # output_N{req}_K{ach}/, sweep_summary.tsv
tasks/                         # todo.md, lessons.md
```
