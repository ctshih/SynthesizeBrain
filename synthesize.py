"""SynthesizeBrain CLI.

Subcommands:
  index       Scan the warp directory, build per-neuron cache. Run once.
  synthesize  Pick N neurons under the packing constraints and emit
              intensity + label volumes (.am + .nii.gz) plus neuron_list.tsv.
  sweep       Run synthesize for several N values in one go.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from synthesize_brain.amira_io import write_ushort_amira
from synthesize_brain.compose import compose
from synthesize_brain.index import build_index, load_index
from synthesize_brain.mip import write_mip
from synthesize_brain.nifti_io import write_nifti
from synthesize_brain.select import select


DEFAULT_WARP_DIR = Path(r"C:\Users\USER\Work\Kaleido\warp")
DEFAULT_CACHE = Path("cache/warp_index.npz")
DEFAULT_OUT_ROOT = Path("output")


def cmd_index(args: argparse.Namespace) -> None:
    build_index(
        warp_dir=args.warp_dir,
        cache_path=args.cache,
        n_workers=args.workers,
        limit=args.limit,
    )


def cmd_synthesize(args: argparse.Namespace) -> None:
    cache = load_index(args.cache)
    N = int(args.n)
    out_dir = args.out or (DEFAULT_OUT_ROOT / f"output_N{N}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    warp_dir = args.warp_dir
    canvas_dims = tuple(int(v) for v in cache["canvas_dims"])
    canvas_bbox = tuple(float(v) for v in cache["canvas_bbox"])

    t0 = time.perf_counter()
    sel = select(cache=cache, N=N, warp_dir=warp_dir, cache_dir=args.cache.parent, seed=args.seed)
    t_select = time.perf_counter() - t0

    t0 = time.perf_counter()
    intensity, labels = compose(sel, canvas_dims)
    t_compose = time.perf_counter() - t0

    # Sanity checks.
    nz_int = int((intensity > 0).sum())
    nz_lbl = int((labels > 0).sum())
    assert nz_int == nz_lbl, f"intensity/label non-zero mismatch: {nz_int} vs {nz_lbl}"
    unique_labels = np.unique(labels)
    K = len(sel.indices)
    assert len(unique_labels) == K + 1, (
        f"unique labels {len(unique_labels)} != K+1={K+1}"
    )

    t0 = time.perf_counter()
    voxel_size = tuple(float(v) for v in cache["voxel_size"])
    write_ushort_amira(out_dir / "intensity.am", intensity, canvas_bbox)
    write_ushort_amira(out_dir / "labels.am", labels, canvas_bbox)
    write_nifti(out_dir / "intensity.nii.gz", intensity, canvas_bbox, voxel_size)
    write_nifti(out_dir / "labels.nii.gz", labels, canvas_bbox, voxel_size)
    write_mip(out_dir / "mip.png", intensity, labels, seed=args.seed)
    t_write = time.perf_counter() - t0

    # neuron_list.tsv
    tsv_path = out_dir / "neuron_list.tsv"
    with open(tsv_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["label_id", "filename", "driver", "voxel_count",
                    "bbox_coverage", "origin_ix", "origin_iy", "origin_iz",
                    "lattice_nx", "lattice_ny", "lattice_nz"])
        for label_id, idx in enumerate(sel.indices, start=1):
            i = int(idx)
            w.writerow([
                label_id,
                str(cache["filenames"][i]),
                str(cache["drivers"][i]),
                int(cache["nnz"][i]),
                f"{float(sel.bbox_coverages[label_id-1]):.4f}",
                int(cache["origin_ix"][i]),
                int(cache["origin_iy"][i]),
                int(cache["origin_iz"][i]),
                int(cache["dims"][i, 0]),
                int(cache["dims"][i, 1]),
                int(cache["dims"][i, 2]),
            ])

    print()
    print(f"[synth] requested N={N}, got K={K}")
    print(f"[synth] C1 coverage: mean={sel.bbox_coverages.mean():.3f}, "
          f"min={sel.bbox_coverages.min():.3f}, "
          f"violators={int((sel.bbox_coverages < 0.5).sum())}")
    print(f"[synth] non-zero voxels: {nz_int} "
          f"({100*nz_int/intensity.size:.2f}% of canvas)")
    print(f"[synth] timings: select={t_select:.1f}s, compose={t_compose:.1f}s, "
          f"write={t_write:.1f}s")
    print(f"[synth] written to {out_dir}/")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run synthesize for each N in args.ns; summarise results per run."""
    # Re-use cmd_synthesize's implementation but log per-N into a summary file.
    summary_path = args.out_root / "sweep_summary.tsv"
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for N in args.ns:
        print(f"\n======= N = {N} =======")
        run_args = argparse.Namespace(
            n=N,
            out=args.out_root / f"output_N{N}",
            cache=args.cache,
            warp_dir=args.warp_dir,
            seed=args.seed,
        )
        t_start = time.perf_counter()
        cmd_synthesize(run_args)
        t_elapsed = time.perf_counter() - t_start

        # Collect per-run stats from neuron_list.tsv.
        tsv_path = run_args.out / "neuron_list.tsv"
        coverages = []
        voxel_counts = []
        with open(tsv_path) as fh:
            next(fh)  # header
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                voxel_counts.append(int(parts[3]))
                coverages.append(float(parts[4]))
        coverages = np.array(coverages)
        voxel_counts = np.array(voxel_counts)
        rows.append({
            "N_requested": N,
            "N_achieved": len(coverages),
            "elapsed_s": round(t_elapsed, 1),
            "coverage_mean": round(float(coverages.mean()), 3),
            "coverage_min": round(float(coverages.min()), 3),
            "coverage_violators": int((coverages < 0.5).sum()),
            "total_nz_voxels": int(voxel_counts.sum()),
        })

    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(list(rows[0].keys()))
        for r in rows:
            w.writerow([r[k] for k in rows[0].keys()])
    print(f"\n[sweep] summary written to {summary_path}")
    for r in rows:
        print(f"  N={r['N_requested']:>4}: got {r['N_achieved']:>4}, "
              f"elapsed {r['elapsed_s']:>6.1f}s, "
              f"coverage mean={r['coverage_mean']:.3f} min={r['coverage_min']:.3f} "
              f"viol={r['coverage_violators']}, "
              f"nz={r['total_nz_voxels']}")


def main() -> None:
    p = argparse.ArgumentParser(prog="synthesize", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_idx = sub.add_parser("index", help="Build the warp-file index cache.")
    p_idx.add_argument("--warp-dir", type=Path, default=DEFAULT_WARP_DIR)
    p_idx.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p_idx.add_argument("--workers", type=int, default=None,
                       help="Parallel workers (default: cpu-1).")
    p_idx.add_argument("--limit", type=int, default=None,
                       help="Scan only the first N files (for smoke tests).")
    p_idx.set_defaults(func=cmd_index)

    p_syn = sub.add_parser("synthesize", help="Compose a single N-neuron volume pair.")
    p_syn.add_argument("--n", type=int, required=True)
    p_syn.add_argument("--out", type=Path, default=None)
    p_syn.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p_syn.add_argument("--warp-dir", type=Path, default=DEFAULT_WARP_DIR)
    p_syn.add_argument("--seed", type=int, default=42)
    p_syn.set_defaults(func=cmd_synthesize)

    p_sw = sub.add_parser("sweep", help="Run synthesize for N in 10/50/100/500.")
    p_sw.add_argument("--ns", type=int, nargs="+", default=[10, 50, 100, 500])
    p_sw.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p_sw.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p_sw.add_argument("--warp-dir", type=Path, default=DEFAULT_WARP_DIR)
    p_sw.add_argument("--seed", type=int, default=42)
    p_sw.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
