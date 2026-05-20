# SynthesizeBrain

*English | [繁體中文](README.zh-TW.md)*

Pack warped single-neuron volumes from the FlyCircuit dataset into paired **intensity + instance-label** 3D volumes, intended as training data for **single-neuron auto-segmentation on dense-staining brain images**.

## Usage

### Install

```bash
pip install -r requirements.txt
```

Main dependencies: `numpy` / `scipy` / `nibabel` / `matplotlib` / `Pillow` / `imageio` / `imageio-ffmpeg` / `tqdm`. Python 3.10+.

### Input data

Point the tool at a directory of FlyCircuit warp volumes (one `.am` file per neuron, AmiraMesh BINARY-LITTLE-ENDIAN 2.1).

Default path: `C:\Users\USER\Work\Kaleido\warp\`

The directory should contain filenames like (one file per neuron):

```
Tdc2-F-000000_seg001_warp_volume.am
Trh-F-000123_seg001_warp_volume.am
VGlut-F-500740_seg001_warp_volume.am
fru-F-900058_seg001_warp_volume.am
...
```

Override the path on any subcommand with `--warp-dir <path>`.

### First-time setup (run once)

```bash
python synthesize.py init
```

Scans the warp directory and builds a per-neuron metadata index (lattice dims, bbox, tight non-zero bbox, voxel count) into `cache/warp_index.npz` (~2 MB). About 18 seconds on 23 workers. Every later command reads from this index.

Only re-run if the source data in `Kaleido\warp\` changes.

### Day-to-day commands

```bash
# Synthesize one fresh run (most common). Defaults: N=500, random seed each time.
python synthesize.py synthesize

# Reproduce a specific run by passing its seed.
python synthesize.py synthesize --seed 2454876261

# Generate many training datasets in a loop (different neuron set each time).
for _ in 1 2 3 4 5; do python synthesize.py synthesize; done             # bash / git-bash
1..5 | ForEach-Object { python synthesize.py synthesize }                # PowerShell
for /L %i in (1,1,5) do python synthesize.py synthesize                  # Windows cmd.exe

# Sweep several N values in one go (e.g. for ceiling testing).
python synthesize.py sweep --ns 10 50 100 500

# Regenerate contacts.csv on an existing output dir.
python synthesize.py contacts --dir output/output_N500_K148_R150_s.../

# Regenerate scan_video.mp4 on an existing output dir.
python synthesize.py video --dir output/output_N500_K148_R150_s.../

# Add or re-roll Gaussian noise on an existing output dir.
python synthesize.py noise --dir output/output_N500_K148_R150_s.../ \
                           --sigma 50 --baseline 100
```

Typical workflow is just `init` once + `synthesize` repeatedly. The other subcommands are alternatives picked based on intent (reproduce, batch sweep, retrofit). You don't chain them.

### Full parameter reference

#### `init` — build the cache (one-time)

| Flag | Default | Meaning |
|---|---|---|
| `--warp-dir` | `C:\Users\USER\Work\Kaleido\warp` | Directory of warp `.am` files |
| `--cache` | `cache/warp_index.npz` | Where to write the cache |
| `--workers` | cpu_count − 1 | Parallel workers |
| `--limit` | all | Scan only the first N files (smoke test) |

#### `synthesize` — compose one paired volume

| Flag | Default | Meaning |
|---|---|---|
| `--n` | `500` | Requested neuron count. Actual K may be smaller when packing saturates |
| `--seed` | random | Random seed. If omitted, a fresh seed is drawn, logged, and embedded in the output dir name for reproducibility |
| `--out` | auto-named | Override output directory (default `output/output_N{req}_K{c1}_R{total}_s{seed}/`) |
| `--cache` | `cache/warp_index.npz` | Read-from cache |
| `--warp-dir` | same as init | Directory of warp `.am` files |
| `--rand-sigma` | `0.25` | Score-noise stdev as a fraction of score std (0 = fully deterministic) |
| `--coverage-threshold` | `0.5` | C1 bbox-coverage threshold in [0, 1]. Lower = looser packing, more neurons survive repair. `0` disables C1 entirely |
| `--score-mode` | `density` | Greedy ordering. `density` = densest brain regions first (default; visually clustered). `small-first` = thinnest neurons first (~30% more neurons; see "Maximizing K"). `hybrid` = score / nnz |
| `--noise-sigma` | `50` | Gaussian noise stdev added to intensity.am (0 = pristine) |
| `--noise-baseline` | `100` | Constant offset added before noise so the background stays a clean Gaussian (~ sCMOS dark-current level) |
| `--no-video` | off | Skip `scan_video.mp4` generation (saves ~50 s/run, useful in batch mode) |

#### `sweep` — run synthesize at multiple N values

| Flag | Default | Meaning |
|---|---|---|
| `--ns` | `10 50 100 500` | List of N values to run |
| `--out-root` | `output` | Root directory for per-N outputs |
| `--cache` / `--warp-dir` / `--seed` / `--rand-sigma` / `--coverage-threshold` / `--score-mode` / `--noise-sigma` / `--noise-baseline` | same as synthesize | |

#### `contacts` — recompute F/E/V contact stats

| Flag | Default | Meaning |
|---|---|---|
| `--dir` | required | Existing output directory |

#### `video` — regenerate scan_video.mp4

| Flag | Default | Meaning |
|---|---|---|
| `--dir` | required | Existing output directory |
| `--fps` | `3` | Frames per second (~0.33 s per label) |

#### `noise` — add / re-roll Gaussian noise on an existing output

| Flag | Default | Meaning |
|---|---|---|
| `--dir` | required | Existing output directory |
| `--sigma` | `50` | Gaussian σ |
| `--baseline` | `100` | Constant offset |
| `--seed` | `0` | RNG seed |

---

## Output format

Each run produces a directory `output/output_N{req}_K{c1}_R{total}_s{seed}/`. The naming components mean:

- `N` = neurons requested on the command line
- `K` = accepted under C1 (greedy + repair survivors)
- `R` = total rendered = K + expand-phase additions (R ≥ K; the expand pass admits voxel-disjoint candidates even if they violate C1, for denser training volumes)
- `s` = resolved random seed (pass `--seed s` to reproduce)

Directory contents:

- `intensity.am` / `intensity.nii.gz` — uint16, voxel values = original warp intensities where a selected neuron lives, 0 elsewhere + Gaussian noise.
- `labels.am` / `labels.nii.gz` — uint16, per-voxel instance label ID (1..K), 0 = background. Paired 1-to-1 with `intensity.*`.
- `neuron_list.tsv` — `label_id, filename, driver, phase, voxel_count, bbox_coverage, origin_{ix,iy,iz}, lattice_{nx,ny,nz}`. The `phase` column is `greedy` / `repair` / `expand` — see "Selection phases" below.
- `contacts.csv` — pairwise F/E/V voxel-touch counts (see "Pairwise contact statistics" below).
- `mip.png` — three-axis MIP preview (top row grayscale intensity; bottom row colorized labels).
- `scan_video.mp4` — per-label scan video (newest neuron bright white, older ones dim grey, top-left caption `N = i`).

## Packing constraints

The selected `N` neurons satisfy:

1. **BBox coverage ≥ threshold** (C1). For each selected neuron *n*, `|bbox(n) ∩ ⋃_{m≠n} bbox(m)| / |bbox(n)| ≥ T`, where `T` defaults to `0.5` and is configurable via `--coverage-threshold` (voxels counted once). Prevents trivially-separated training cases. Set `T=0` to disable C1 (everything just needs C2).
2. **Voxel-level disjointness** (C2). Selected neurons' non-zero voxels are pairwise disjoint (touching is fine, overlap is not). Real dense tissue doesn't overlap; warps from different flies can.

A single synthesis can mix drivers (Tdc2 / Trh / VGlut / fru).

## Selection phases

The selector runs in three phases, and each neuron's phase is recorded in `neuron_list.tsv → phase`:

1. **greedy** — pack candidates in decreasing-score order until N is reached or the candidate pool is exhausted. Enforces voxel-disjointness (C2); does *not* yet check C1.
2. **repair** — validate C1 per neuron; drop violators (free their voxels and bbox contribution), refill from the remaining pool. Loops up to 5 rounds. Survivors all satisfy C1 by the end.
3. **expand** — once no more neurons can replace violators, sweep the remaining pool once more and admit any that still fit by voxel disjointness alone, *ignoring* C1. These are training-density bonuses; they don't carry the C1 guarantee.

## Maximizing K (when you want as many neurons as possible)

The default packing saturates around K ≈ 140–180 because the greedy phase places **dense, voxel-heavy neurons first**, blocking thinner ones from fitting later. To pack more neurons:

```bash
python synthesize.py synthesize --n 500 --score-mode small-first --coverage-threshold 0
```

`small-first` orders the greedy phase by neuron size ascending — thin neurons fit into the canvas gaps left by fat ones. Combined with `--coverage-threshold 0` (disable C1 repair churn), this typically yields **~30% more neurons** at the cost of a slightly lower mean bbox-coverage (e.g. mean 0.86 → 0.83). Same-seed comparison on the 28 620-neuron dataset:

| `--score-mode` | `--coverage-threshold` | K | R | mean coverage |
|---|---|---|---|---|
| `density` (default) | `0.5` | 147 | 148 | 0.860 |
| `small-first` | `0.5` | 191 | 195 | 0.838 |
| `small-first` | `0.0` | **193** | 193 | 0.832 |
| `hybrid` | `0.5` | 169 | 173 | 0.844 |

The trade-off: more neurons, but lower **total non-zero voxel count** (small neurons = less mass each) and a longer tail of low-coverage neurons (e.g. 7 neurons below 0.5, vs 1 in `density`). Pick `small-first` if your training cares about instance diversity, `density` if it cares about realistic dense-staining appearance.

## Gaussian noise on intensity.am

Model: `intensity' = intensity + baseline + N(0, σ)`, clipped to `[0, 65535]`.

Defaults: σ = 50, baseline = 100. The **baseline offset** keeps the background a clean Gaussian (mean = baseline) instead of the folded / half-Gaussian you would get from clipping at 0 — matching real sCMOS / CCD cameras that always report a positive dark-current offset.

Empirical check: at σ = 50 / baseline = 100, a 10×10×10 background corner reports mean ≈ 97.8, std ≈ 50.2 — matching N(100, 50) within sampling noise. `labels.am` is never noised — ground truth stays crisp.

## Inspecting individual labels in Avizo

Open `labels.am` (label field) + `colormaps/bandpass_white.am` in Avizo, then on the colormap port set `MinMax` to:

| Goal | MinMax |
|---|---|
| only label N | `[N - 0.5, N + 0.5]` |
| labels A..B inclusive | `[A - 0.5, B + 0.5]` |
| scan one label at a time | `[0.5, 1.5]` → `[1.5, 2.5]` → ... → `[K - 0.5, K + 0.5]` |

**Why the half-unit offset**: Avizo treats `[N, N]` (zero-width) as an empty range and shows nothing — a half-unit on each side is what brackets the integer label N. Thin neurons may still be hard to see; switch Volume Rendering Composition to `max` (MIP mode), or use Ortho Slice for a cleaner view.

## Pairwise contact statistics

`contacts.csv` columns: `neuron1, neuron2, N_F, N_E, N_V`. Only pairs with at least one contact voxel are listed:

- **F** — face-adjacent (share a full face of their unit voxel cubes)
- **E** — edge-adjacent only (share an edge but not a face)
- **V** — vertex-adjacent only (share a corner vertex but not an edge)

## Empirical packing ceiling

With the dataset in `Kaleido\warp\` (9987 FlyCircuit warps):

| N requested | K (under C1) | R (total) | notes |
|------------:|-------------:|----------:|-------|
| 10          | 10           | 10        | C1 unstressed |
| 50          | 50           | 50        | C1 OK |
| 100         | 100          | 100       | C1 OK |
| 500         | ~130–150     | ~135–160  | varies per seed; voxel packing saturates |

Voxel-disjoint packing saturates around **R ≈ 140–160** — past that, no remaining neuron fits without overlap. Each random seed explores a different local optimum, so repeated runs at N=500 give different K/R values and different neuron subsets — the intended behaviour for generating many training datasets.

11 runs × N=500 in practice: 1079 unique neurons total, mean pairwise Jaccard 0.06, 70% of unique neurons appear in only one run — strong training-data diversity.

## Algorithm sketch

- **Phase 1 — `synthesize_brain/index.py`**: scan every `.am`; record cropped lattice dims, header bbox, **tight non-zero bbox**, non-zero voxel count. Canvas = union of all header bboxes, rounded to integer voxels.
- **Phase 2 — `synthesize_brain/select.py`**:
  1. Build a canvas-voxel coverage counter across the full dataset; score every neuron by mean coverage inside its own tight bbox (central = high).
  2. Greedy accept candidates in decreasing score order (tie-break: smaller neurons first), rejecting any whose voxels intersect `occupied_mask`. Reads are parallelised across 8 threads with a bounded prefetch window; rejected reads are retained in a shared in-memory cache so the repair pass does not re-decode them.
  3. Validate constraint (1) using an incrementally-maintained per-voxel bbox counter; drop violators, refill from the score-sorted pool, iterate up to 5 rounds.
- **Phase 3 — `synthesize_brain/compose.py`**: paste each accepted neuron's non-zero voxels into `intensity` and write its label ID into `labels`.
- **Phase 4 — writers**: `amira_io.py` (raw AmiraMesh ushort + label field), `nifti_io.py` (`.nii.gz` via nibabel), `mip.py` (three-axis MIP PNG), `contacts.py` (pairwise F/E/V voxel-touch counts), `scan_video.py` (per-label scan MP4), `noise.py` (Gaussian noise).

## Repo layout

```
synthesize.py                  # CLI entry point
synthesize_brain/
├── amira_io.py                # AmiraMesh reader (vendored from Kaleido) + ushort / label writer
├── compose.py                 # Phase 3
├── contacts.py                # Pairwise F/E/V contact counts
├── index.py                   # Phase 1 (implementation of the `init` subcommand)
├── mip.py                     # MIP preview
├── nifti_io.py                # NIfTI writer
├── noise.py                   # Gaussian noise
├── scan_video.py              # Per-label scan MP4
└── select.py                  # Phase 2
colormaps/bandpass_white.am    # Bandpass colormap for Avizo per-label inspection
check_saturation.py            # Diagnostic: is the final K truly voxel-saturated?
analyze_overlap.py             # Across-run neuron-overlap diversity report
cache/                         # warp_index.npz, candidate_scores.npz
output/                        # output_N{req}_K{c1}_R{total}_s{seed}/, sweep_summary.tsv
tasks/                         # todo.md, lessons.md
docs/                          # PPTX deck, tutorial, build script
```
