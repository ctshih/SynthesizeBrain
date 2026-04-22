"""Diagnostic: at the final N=500→K=148 state, can any remaining neuron
still be packed in without voxel overlap? If yes, the pipeline stopped
early; if no, 148 is a true voxel-packing saturation point (ignoring C1).
"""

from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from synthesize_brain.amira_io import read_amira
from synthesize_brain.index import load_index


OUT_DIR = Path("output/output_N500_K148")
WARP_DIR = Path(r"C:\Users\USER\Work\Kaleido\warp")
CACHE = Path("cache/warp_index.npz")


def load_selected_names(tsv_path: Path) -> set[str]:
    names = set()
    with open(tsv_path) as fh:
        r = csv.reader(fh, delimiter="\t")
        next(r)
        for row in r:
            names.add(row[1])
    return names


def rebuild_occupied_mask(selected_idx: list[int], cache: dict, warp_dir: Path) -> np.ndarray:
    nx, ny, nz = [int(v) for v in cache["canvas_dims"]]
    occupied = np.zeros((nz, ny, nx), dtype=bool)
    for i in tqdm(selected_idx, desc="rebuild occupied"):
        name = str(cache["filenames"][i])
        vol = read_amira(warp_dir / name)
        nzi = np.argwhere(vol.data > 0)
        zs = nzi[:, 0] + int(cache["origin_iz"][i])
        ys = nzi[:, 1] + int(cache["origin_iy"][i])
        xs = nzi[:, 2] + int(cache["origin_ix"][i])
        occupied[zs, ys, xs] = True
    return occupied


def fits_into(i: int, cache: dict, warp_dir: Path, occupied: np.ndarray) -> bool:
    name = str(cache["filenames"][i])
    try:
        vol = read_amira(warp_dir / name)
    except Exception:
        return False
    nzi = np.argwhere(vol.data > 0)
    if nzi.size == 0:
        return False
    zs = nzi[:, 0] + int(cache["origin_iz"][i])
    ys = nzi[:, 1] + int(cache["origin_iy"][i])
    xs = nzi[:, 2] + int(cache["origin_ix"][i])
    return not bool(occupied[zs, ys, xs].any())


def main() -> None:
    cache = load_index(CACHE)
    selected_names = load_selected_names(OUT_DIR / "neuron_list.tsv")
    all_names = list(cache["filenames"])
    name_to_idx = {str(n): i for i, n in enumerate(all_names)}

    selected_idx = sorted(name_to_idx[n] for n in selected_names)
    remaining_idx = [i for i, n in enumerate(all_names) if str(n) not in selected_names]
    print(f"selected K = {len(selected_idx)}, remaining = {len(remaining_idx)}")

    t = time.perf_counter()
    occupied = rebuild_occupied_mask(selected_idx, cache, WARP_DIR)
    print(f"occupied voxels = {int(occupied.sum())}, rebuild in {time.perf_counter()-t:.1f}s")

    # Parallel fits check.
    fits_any = []
    t = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fits_into, i, cache, WARP_DIR, occupied) for i in remaining_idx]
        for idx, fut in zip(remaining_idx, tqdm(futs, desc="test fit")):
            if fut.result():
                fits_any.append(idx)
    print(f"candidates that STILL fit (no voxel overlap): {len(fits_any)}  "
          f"(elapsed {time.perf_counter()-t:.1f}s)")

    if fits_any:
        print("\nExamples of neurons that could still be packed in:")
        # Show top 10 by nnz, since bigger = more interesting.
        fits_sorted = sorted(fits_any, key=lambda i: -int(cache["nnz"][i]))
        for i in fits_sorted[:10]:
            print(f"  idx={i:>5}  nnz={int(cache['nnz'][i]):>7}  {str(cache['filenames'][i])}")


if __name__ == "__main__":
    main()
