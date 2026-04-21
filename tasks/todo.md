## Plan: SynthesizeBrain — paired intensity + instance-label volumes

### Commit 1 — skeleton + Phase 1 indexer
- [x] Project skeleton (`synthesize_brain/` package, `cache/`, `output/`, `tasks/`)
- [x] Vendor AmiraMesh reader from Kaleido into `synthesize_brain/amira_io.py`; add ushort writer (raw)
- [x] `synthesize_brain/index.py`: scan every warp `.am`, record driver / dims / bbox / tight non-zero bbox / nnz; derive canvas; save `.npz`
- [x] CLI `synthesize.py` with `index`, `synthesize`, `sweep` subcommands (only `index` implemented in this commit)
- [ ] Smoke test `index` on `--limit 50` then run on all 9987 files; time it

### Commit 2 — selection + synthesis
- [x] `synthesize_brain/select.py`:
  - Build bbox-coverage counter map over canvas → score candidates
  - Cache scores to `cache/candidate_scores.npz` (skip ~1-min rebuild per run)
  - Greedy pick by decreasing coverage score, parallel prefetch (8 threads)
  - Shared in-memory read cache so repair-round refills don't re-decode
  - Verify constraint 1 via incrementally-maintained bbox counter; swap offenders
- [x] `synthesize_brain/compose.py`: deposit intensity + label for each selected neuron
- [x] Wire to `cmd_synthesize`; dump `neuron_list.tsv`

### Commit 3 — outputs + multi-N sweep
- [x] `synthesize_brain/nifti_io.py`: `.nii.gz` writer via nibabel (affine = spacing/origin)
- [x] `synthesize_brain/mip.py`: three-axis MIP PNG (grey intensity + random-colored labels)
- [x] `cmd_sweep`: run N ∈ {10, 50, 100, 500}, write `sweep_summary.tsv`
- [x] README with usage

### Review
- **What changed**: new standalone project. Index → select → compose → write pipeline. Four writers (AmiraMesh ushort raw, NIfTI .nii.gz, MIP PNG, TSV). Parallel prefetch + in-memory read cache in the selection phase.
- **Why**: produce paired (intensity, instance-label) training volumes under two geometric constraints (bbox coverage ≥50%, voxel-level disjointness). Targeted at single-neuron auto-segmentation training for dense-staining brain images.
- **Lessons learned**:
  1. Per-neuron `.am` files are tightly cropped with per-file BoundingBox — not one uniform lattice. Shared standard-brain canvas is 989×646×337 (union of all 9987 per-file bboxes), not 386×345×182.
  2. When adding parallel I/O, **profile the whole function**, not just the parallel part — our 3.7× speedup of greedy fill was hidden by a serial refill loop that re-read the same 9000 candidates. An in-memory read cache during selection fixed this at zero cost.
  3. Empirical packing ceiling is K ≈ 148 — the user's hypothesis that N >> 100 would be infeasible was right. Voxel-disjoint packing saturates long before we run out of neurons.
