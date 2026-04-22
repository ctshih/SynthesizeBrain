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

from synthesize_brain.amira_io import read_amira, write_label_amira, write_ushort_amira
from synthesize_brain.compose import compose
from synthesize_brain.contacts import write_contacts_csv
from synthesize_brain.index import build_index, load_index
from synthesize_brain.mip import random_label_colors, write_mip
from synthesize_brain.nifti_io import write_nifti
from synthesize_brain.scan_video import write_scan_video
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


def _run_one(
    N: int,
    cache: dict,
    cache_dir: Path,
    warp_dir: Path,
    seed: int | None,
    rand_sigma: float,
    out_root: Path,
    out_override: Path | None = None,
) -> dict:
    """Run selection + synthesis for a single N. Returns a stats dict.

    Output directory naming: if `out_override` is given we respect it
    verbatim; otherwise we build
    `{out_root}/output_N{req}_K{c1}_R{total}_s{seed}/` — where
        N = neurons requested
        K = neurons accepted under C1 (greedy + repair survivors)
        R = total rendered (K + expand-phase additions; R >= K)
    Pass `--seed {seed}` back to reproduce the exact neuron set.
    """
    canvas_dims = tuple(int(v) for v in cache["canvas_dims"])
    canvas_bbox = tuple(float(v) for v in cache["canvas_bbox"])
    voxel_size = tuple(float(v) for v in cache["voxel_size"])

    t0 = time.perf_counter()
    sel = select(cache=cache, N=N, warp_dir=warp_dir, cache_dir=cache_dir,
                 seed=seed, rand_sigma=rand_sigma)
    t_select = time.perf_counter() - t0

    t0 = time.perf_counter()
    intensity, labels = compose(sel, canvas_dims)
    t_compose = time.perf_counter() - t0

    nz_int = int((intensity > 0).sum())
    nz_lbl = int((labels > 0).sum())
    assert nz_int == nz_lbl, f"intensity/label non-zero mismatch: {nz_int} vs {nz_lbl}"
    R = len(sel.indices)                                  # total rendered
    K = int(sum(p != "expand" for p in sel.phases))       # under-C1 subset
    unique_labels = np.unique(labels)
    assert len(unique_labels) == R + 1, f"unique labels {len(unique_labels)} != R+1={R+1}"

    out_dir = (
        Path(out_override)
        if out_override is not None
        else (out_root / f"output_N{N}_K{K}_R{R}_s{sel.seed_used}")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shared label palette: the AmiraMesh `Materials` block and the MIP PNG
    # use the same colors, so what you see in Avizo == what you see in mip.png.
    label_palette = random_label_colors(R, seed=sel.seed_used)

    # Build per-label names that include the driver so Avizo's Materials panel
    # is self-describing ("VGlut_500740" instead of anonymous "Material3").
    label_names = ["Exterior"]
    for idx in sel.indices:
        i = int(idx)
        raw = str(cache["filenames"][i])
        # e.g. "VGlut-F-500740_seg001_warp_volume.am" -> "VGlut_F_500740"
        stem = raw.split("_seg")[0].replace("-", "_")
        label_names.append(stem)

    t0 = time.perf_counter()
    write_ushort_amira(out_dir / "intensity.am", intensity, canvas_bbox)
    write_label_amira(out_dir / "labels.am", labels, canvas_bbox,
                      label_colors=label_palette, label_names=label_names)
    write_nifti(out_dir / "intensity.nii.gz", intensity, canvas_bbox, voxel_size)
    write_nifti(out_dir / "labels.nii.gz", labels, canvas_bbox, voxel_size)
    write_mip(out_dir / "mip.png", intensity, labels, seed=sel.seed_used)
    t_write = time.perf_counter() - t0

    tsv_path = out_dir / "neuron_list.tsv"
    label_to_name: dict[int, str] = {}
    with open(tsv_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["label_id", "filename", "driver", "phase", "voxel_count",
                    "bbox_coverage", "origin_ix", "origin_iy", "origin_iz",
                    "lattice_nx", "lattice_ny", "lattice_nz"])
        for label_id, idx in enumerate(sel.indices, start=1):
            i = int(idx)
            fname = str(cache["filenames"][i])
            label_to_name[label_id] = fname
            w.writerow([
                label_id,
                fname,
                str(cache["drivers"][i]),
                str(sel.phases[label_id - 1]),
                int(cache["nnz"][i]),
                f"{float(sel.bbox_coverages[label_id-1]):.4f}",
                int(cache["origin_ix"][i]),
                int(cache["origin_iy"][i]),
                int(cache["origin_iz"][i]),
                int(cache["dims"][i, 0]),
                int(cache["dims"][i, 1]),
                int(cache["dims"][i, 2]),
            ])

    # Pairwise F/E/V contact statistics.
    t0 = time.perf_counter()
    n_pairs = write_contacts_csv(out_dir / "contacts.csv", labels, label_to_name)
    t_contacts = time.perf_counter() - t0
    print(f"[synth] contacts: {n_pairs} touching pairs written in {t_contacts:.1f}s")

    # Per-label scan video — one frame per label, newest white, seen dim,
    # with a "N=i" caption. Handy for quickly eyeballing each label.
    t0 = time.perf_counter()
    write_scan_video(labels, out_dir / "scan_video.mp4", verbose=False)
    t_video = time.perf_counter() - t0
    print(f"[synth] scan_video.mp4 written in {t_video:.1f}s")

    print()
    print(f"[synth] requested N={N}, kept under C1 K={K}, total rendered R={R}")
    print(f"[synth] C1 coverage: mean={sel.bbox_coverages.mean():.3f}, "
          f"min={sel.bbox_coverages.min():.3f}, "
          f"violators={int((sel.bbox_coverages < 0.5).sum())}")
    print(f"[synth] non-zero voxels: {nz_int} "
          f"({100*nz_int/intensity.size:.2f}% of canvas)")
    print(f"[synth] timings: select={t_select:.1f}s, compose={t_compose:.1f}s, "
          f"write={t_write:.1f}s")
    print(f"[synth] written to {out_dir}/")

    return {
        "N_requested": N,
        "K_under_c1": K,
        "R_total": R,
        "out_dir": str(out_dir),
        "t_select": t_select,
        "t_compose": t_compose,
        "t_write": t_write,
        "coverage_mean": float(sel.bbox_coverages.mean()),
        "coverage_min": float(sel.bbox_coverages.min()),
        "coverage_violators": int((sel.bbox_coverages < 0.5).sum()),
        "total_nz_voxels": nz_int,
    }


def cmd_synthesize(args: argparse.Namespace) -> None:
    cache = load_index(args.cache)
    _run_one(
        N=int(args.n),
        cache=cache,
        cache_dir=args.cache.parent,
        warp_dir=args.warp_dir,
        seed=args.seed,
        rand_sigma=args.rand_sigma,
        out_root=DEFAULT_OUT_ROOT,
        out_override=args.out,
    )


def cmd_video(args: argparse.Namespace) -> None:
    """Regenerate scan_video.mp4 for an existing output directory."""
    out_dir = Path(args.dir)
    labels_path = out_dir / "labels.am"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.am not found in {out_dir}")
    labels = read_amira(labels_path).data
    out_path = out_dir / "scan_video.mp4"
    t0 = time.perf_counter()
    write_scan_video(labels, out_path, fps=args.fps)
    print(f"[video] wrote {out_path} in {time.perf_counter()-t0:.1f}s")


def cmd_contacts(args: argparse.Namespace) -> None:
    """Recompute contacts.csv on an existing output dir.

    Useful for applying contact analysis to volumes synthesized before this
    feature existed, or tweaking output format without re-running selection.
    """
    out_dir = Path(args.dir)
    labels_path = out_dir / "labels.am"
    tsv_path = out_dir / "neuron_list.tsv"
    if not labels_path.exists() or not tsv_path.exists():
        raise FileNotFoundError(f"need labels.am and neuron_list.tsv in {out_dir}")

    labels = read_amira(labels_path).data  # (Z, Y, X) uint16
    label_to_name: dict[int, str] = {}
    with open(tsv_path) as fh:
        r = csv.reader(fh, delimiter="\t")
        header = next(r)
        i_id = header.index("label_id")
        i_fn = header.index("filename")
        for row in r:
            label_to_name[int(row[i_id])] = row[i_fn]

    t0 = time.perf_counter()
    n = write_contacts_csv(out_dir / "contacts.csv", labels, label_to_name)
    print(f"[contacts] wrote {n} pairs to {out_dir/'contacts.csv'} "
          f"in {time.perf_counter()-t0:.1f}s")


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run synthesize for each N in args.ns; summarise results per run."""
    cache = load_index(args.cache)
    args.out_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for N in args.ns:
        print(f"\n======= N = {N} =======")
        t_start = time.perf_counter()
        stats = _run_one(
            N=N,
            cache=cache,
            cache_dir=args.cache.parent,
            warp_dir=args.warp_dir,
            seed=args.seed,
            rand_sigma=args.rand_sigma,
            out_root=args.out_root,
        )
        stats["elapsed_s"] = round(time.perf_counter() - t_start, 1)
        rows.append(stats)

    summary_path = args.out_root / "sweep_summary.tsv"
    cols = ["N_requested", "K_under_c1", "R_total", "elapsed_s", "coverage_mean",
            "coverage_min", "coverage_violators", "total_nz_voxels", "out_dir"]
    with open(summary_path, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols)
        for r in rows:
            w.writerow([
                r["N_requested"], r["K_under_c1"], r["R_total"], r["elapsed_s"],
                round(r["coverage_mean"], 3), round(r["coverage_min"], 3),
                r["coverage_violators"], r["total_nz_voxels"], r["out_dir"],
            ])
    print(f"\n[sweep] summary written to {summary_path}")
    for r in rows:
        print(f"  N={r['N_requested']:>4}: K={r['K_under_c1']:>4} R={r['R_total']:>4}, "
              f"elapsed {r['elapsed_s']:>6.1f}s, "
              f"coverage mean={r['coverage_mean']:.3f} min={r['coverage_min']:.3f} "
              f"viol={r['coverage_violators']}, "
              f"nz={r['total_nz_voxels']}  -> {r['out_dir']}")


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
    p_syn.add_argument("--n", type=int, default=500,
                       help="Requested neuron count (default: 500). "
                            "The actual K may be smaller when packing saturates.")
    p_syn.add_argument("--out", type=Path, default=None,
                       help="Override the auto-named output dir (default: "
                            "output/output_N{req}_K{ach}_s{seed}/).")
    p_syn.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p_syn.add_argument("--warp-dir", type=Path, default=DEFAULT_WARP_DIR)
    p_syn.add_argument("--seed", type=int, default=None,
                       help="Random seed. If omitted, a fresh seed is drawn "
                            "so repeated runs yield different neuron sets "
                            "(the resolved seed is logged and embedded in "
                            "the output dir name for reproducibility).")
    p_syn.add_argument("--rand-sigma", type=float, default=0.25,
                       help="Score-noise stdev as a fraction of score std "
                            "(0 disables randomness; default 0.25).")
    p_syn.set_defaults(func=cmd_synthesize)

    p_ct = sub.add_parser("contacts",
                          help="Compute pairwise F/E/V contacts on an existing output dir.")
    p_ct.add_argument("--dir", type=Path, required=True,
                      help="Path to an output_N{...}_K{...}/ directory.")
    p_ct.set_defaults(func=cmd_contacts)

    p_vid = sub.add_parser("video",
                           help="Regenerate scan_video.mp4 for an existing output dir.")
    p_vid.add_argument("--dir", type=Path, required=True,
                       help="Path to an output_N{...}_K{...}/ directory.")
    p_vid.add_argument("--fps", type=float, default=3.0,
                       help="Frames per second (default: 3, ~0.33s per label).")
    p_vid.set_defaults(func=cmd_video)

    p_sw = sub.add_parser("sweep", help="Run synthesize for several N values.")
    p_sw.add_argument("--ns", type=int, nargs="+", default=[10, 50, 100, 500])
    p_sw.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    p_sw.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p_sw.add_argument("--warp-dir", type=Path, default=DEFAULT_WARP_DIR)
    p_sw.add_argument("--seed", type=int, default=None)
    p_sw.add_argument("--rand-sigma", type=float, default=0.25)
    p_sw.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
