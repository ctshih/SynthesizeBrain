"""Phase 2: select N neurons satisfying the packing constraints.

Constraints:
  (1) Each selected neuron's tight bbox must have >=50% of its volume covered
      by the union of the other selected neurons' tight bboxes (voxel counted
      at most once — this is |bbox(n) ∩ ⋃_{m≠n} bbox(m)| / |bbox(n)|).
  (2) Non-zero voxels of selected neurons must be disjoint across the set.

Algorithm:
  A. Build an int16 coverage counter over the canvas — count how many tight
     bboxes across the ENTIRE dataset cover each voxel. This is a coarse
     proxy for "brain density"; central voxels have high count.
  B. Score every candidate neuron by the mean coverage-count inside its own
     tight bbox. High score = sits in a dense region = likely to satisfy C1.
  C. Greedy selection in decreasing score order:
       - Fast path: reject if candidate's tight bbox is entirely outside the
         current bounding hull (no chance of helping C1 early on — skip
         after a few are already picked if too far).
       - Load the candidate's volume; check voxel-level disjointness vs
         occupied_mask; if clean, add and update masks.
  D. After N candidates are accepted, verify C1 per-neuron; drop violators
     (free their voxels from occupied_mask) and attempt to refill from the
     remaining candidate pool.
  E. Iterate D for a few rounds; stop when stable or max iterations reached.
"""

from __future__ import annotations

import collections
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .amira_io import read_amira


_SCORE_CACHE_FILENAME = "candidate_scores.npz"


@dataclass
class SelectionResult:
    indices: np.ndarray           # (K,) indices into the cache
    bbox_coverages: np.ndarray    # (K,) per-neuron C1 coverage ratios
    phases: np.ndarray            # (K,) str — "greedy" / "repair" / "expand"
    occupied_mask: np.ndarray     # (Z, Y, X) bool — union of selected voxel masks
    volumes: list[tuple[int, np.ndarray, tuple[int, int, int]]]
    # volumes: list of (cache_idx, nonzero_values_uint16, (zs, ys, xs)_canvas_indices)
    # stored here so compose.py can paste them without re-reading from disk
    seed_used: int = 0            # the actual seed value used (for reproducibility)


def _tight_bbox_slices(idx: int, cache: dict) -> tuple[slice, slice, slice]:
    zs = slice(int(cache["tight_iz_min"][idx]), int(cache["tight_iz_max"][idx]) + 1)
    ys = slice(int(cache["tight_iy_min"][idx]), int(cache["tight_iy_max"][idx]) + 1)
    xs = slice(int(cache["tight_ix_min"][idx]), int(cache["tight_ix_max"][idx]) + 1)
    return zs, ys, xs


def _build_coverage_map(cache: dict, canvas_dims: tuple[int, int, int]) -> np.ndarray:
    """For every canvas voxel, count how many tight bboxes across the full
    dataset contain it. Returns int16 (saturates at 32767 — more than enough)."""
    nx, ny, nz = canvas_dims
    cov = np.zeros((nz, ny, nx), dtype=np.int16)
    N = len(cache["filenames"])
    for i in tqdm(range(N), desc="coverage map"):
        zs, ys, xs = _tight_bbox_slices(i, cache)
        cov[zs, ys, xs] += 1
    return cov


def _candidate_scores(cache: dict, cov: np.ndarray) -> np.ndarray:
    """For each neuron: mean coverage-count inside its tight bbox."""
    N = len(cache["filenames"])
    scores = np.empty(N, dtype=np.float32)
    for i in range(N):
        zs, ys, xs = _tight_bbox_slices(i, cache)
        sub = cov[zs, ys, xs]
        scores[i] = float(sub.mean()) if sub.size else 0.0
    return scores


def _load_neuron_canvas_voxels(
    idx: int,
    cache: dict,
    warp_dir: Path,
    read_cache: dict | None = None,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read the neuron's .am, return (nonzero_values_uint16, (zs_canvas, ys_canvas, xs_canvas)).

    If `read_cache` is provided, results are memoised by `idx` so the refill
    pass can skip re-reading candidates the greedy pass already touched.
    """
    if read_cache is not None and idx in read_cache:
        return read_cache[idx]
    name = str(cache["filenames"][idx])
    path = warp_dir / name
    vol = read_amira(path)
    data = vol.data  # (Z, Y, X) uint16
    nz_idx = np.argwhere(data > 0)  # (K, 3) local (z, y, x)
    values = data[nz_idx[:, 0], nz_idx[:, 1], nz_idx[:, 2]]
    ox = int(cache["origin_ix"][idx])
    oy = int(cache["origin_iy"][idx])
    oz = int(cache["origin_iz"][idx])
    zs = nz_idx[:, 0].astype(np.int32) + oz
    ys = nz_idx[:, 1].astype(np.int32) + oy
    xs = nz_idx[:, 2].astype(np.int32) + ox
    result = (values, (zs, ys, xs))
    if read_cache is not None:
        read_cache[idx] = result
    return result


def _prefetched_reads(
    order,
    cache: dict,
    warp_dir: Path,
    read_cache: dict | None = None,
    n_workers: int = 8,
    max_inflight: int = 32,
):
    """Yield (idx, values, (zs,ys,xs)) for every idx in `order`, reading ahead
    in `n_workers` background threads. zlib decompress + numpy releases the GIL,
    so threads effectively parallelise the HxZip read."""
    it = iter(order)
    ex = ThreadPoolExecutor(max_workers=n_workers)
    pending: collections.deque = collections.deque()
    try:
        for _ in range(max_inflight):
            try:
                idx = next(it)
            except StopIteration:
                break
            pending.append((int(idx), ex.submit(_load_neuron_canvas_voxels, int(idx), cache, warp_dir, read_cache)))
        while pending:
            idx, fut = pending.popleft()
            try:
                values, idx_tuple = fut.result()
            except Exception:
                values, idx_tuple = None, None
            yield idx, values, idx_tuple
            try:
                next_idx = next(it)
            except StopIteration:
                continue
            pending.append((int(next_idx), ex.submit(_load_neuron_canvas_voxels, int(next_idx), cache, warp_dir, read_cache)))
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


def _init_bbox_counter(
    selected: list[int],
    cache: dict,
    canvas_dims: tuple[int, int, int],
) -> np.ndarray:
    """Allocate and fill the per-voxel count of how many SELECTED bboxes cover it."""
    nx, ny, nz = canvas_dims
    counter = np.zeros((nz, ny, nx), dtype=np.int16)
    for i in selected:
        zs, ys, xs = _tight_bbox_slices(i, cache)
        counter[zs, ys, xs] += 1
    return counter


def _bbox_counter_add(counter: np.ndarray, idx: int, cache: dict) -> None:
    zs, ys, xs = _tight_bbox_slices(idx, cache)
    counter[zs, ys, xs] += 1


def _bbox_counter_remove(counter: np.ndarray, idx: int, cache: dict) -> None:
    zs, ys, xs = _tight_bbox_slices(idx, cache)
    counter[zs, ys, xs] -= 1


def _coverage_ratios(
    selected: list[int],
    counter: np.ndarray,
    cache: dict,
) -> np.ndarray:
    """For each selected n: fraction of bbox(n) covered by >=1 OTHER selected bbox."""
    ratios = np.empty(len(selected), dtype=np.float32)
    for k, i in enumerate(selected):
        zs, ys, xs = _tight_bbox_slices(i, cache)
        sub = counter[zs, ys, xs]
        # counter at voxel = (this bbox's contribution=1) + (others covering it).
        # sub >= 2 ⇔ at least one OTHER selected bbox covers this voxel.
        covered_by_others = int((sub >= 2).sum())
        ratios[k] = covered_by_others / max(sub.size, 1)
    return ratios


def _scores_cache_path(cache_dir: Path) -> Path:
    return cache_dir / _SCORE_CACHE_FILENAME


def _compute_or_load_scores(
    cache: dict,
    canvas_dims: tuple[int, int, int],
    cache_dir: Path,
    verbose: bool = True,
) -> np.ndarray:
    """Candidate scores depend only on the index (not on N or seed), so cache them."""
    sc_path = _scores_cache_path(cache_dir)
    N = len(cache["filenames"])
    if sc_path.exists():
        data = np.load(sc_path)
        if len(data["scores"]) == N and tuple(data["canvas_dims"]) == canvas_dims:
            if verbose:
                print(f"[select] loaded cached scores from {sc_path}")
            return data["scores"].astype(np.float32)

    t0 = time.perf_counter()
    cov = _build_coverage_map(cache, canvas_dims)
    if verbose:
        print(f"[select] coverage map built in {time.perf_counter()-t0:.1f}s, "
              f"max stack = {int(cov.max())} neurons on a single voxel")
    t1 = time.perf_counter()
    scores = _candidate_scores(cache, cov)
    if verbose:
        print(f"[select] scored {N} candidates in {time.perf_counter()-t1:.1f}s")

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(sc_path, scores=scores, canvas_dims=np.array(canvas_dims, dtype=np.int32))
    return scores


def select(
    cache: dict,
    N: int,
    warp_dir: Path,
    cache_dir: Path,
    seed: int | None = None,
    rand_sigma: float = 0.25,
    coverage_threshold: float = 0.5,
    score_mode: str = "density",
    max_repair_rounds: int = 5,
    max_candidates_per_round: int | None = None,
    expand_after_repair: bool = True,
    verbose: bool = True,
) -> SelectionResult:
    """Run Phase 2 selection and return up to N neurons satisfying the packing constraints.

    `seed`: if None, draws a fresh entropy-based seed and logs it so the run
        can be reproduced by passing the same value later.
    `rand_sigma`: Gaussian noise stdev applied to candidate scores, expressed
        as a fraction of the score distribution's stdev (0 disables randomness
        and gives deterministic "densest first" ordering).
    `score_mode`:
        - "density"     : sort by densest-first (default; biases toward central
                          brain regions, packing fewer but more clustered neurons).
        - "small-first" : sort by smallest-nnz first (packs more neurons by
                          using up canvas with thin neurons before fat ones).
        - "hybrid"      : sort by `score / nnz` (per-voxel density), a mix.
    `expand_after_repair`: after the C1-aware repair loop, iterate remaining
        candidates once more and admit any that don't voxel-overlap, EVEN IF
        they violate C1. Gives the richest packing — paired (intensity, label)
        volumes for segmentation benefit from more neurons regardless of C1.
    """
    canvas_dims = tuple(int(v) for v in cache["canvas_dims"])  # (X, Y, Z) declared
    # canvas_dims from cache is (nx, ny, nz); arrays here are (z, y, x).
    nx_c, ny_c, nz_c = canvas_dims
    canvas_zyx = (nz_c, ny_c, nx_c)

    # Resolve seed: draw fresh entropy if None so each unseeded run is different.
    if seed is None:
        seed = int(np.random.SeedSequence().entropy & 0xFFFFFFFF)
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"[select] canvas (z,y,x) = {canvas_zyx}, requesting N={N}, seed={seed}")

    # A + B. Coverage-map + per-neuron scoring. Cache across runs.
    scores = _compute_or_load_scores(cache, canvas_dims, cache_dir, verbose=verbose)

    # Apply seed-driven Gaussian noise so each run picks a different set while
    # staying biased toward dense regions. sigma = rand_sigma × std(scores).
    if rand_sigma > 0:
        noise_std = rand_sigma * float(scores.std())
        perturbed = scores + rng.normal(0.0, noise_std, size=scores.shape).astype(np.float32)
        if verbose:
            print(f"[select] score noise σ = {noise_std:.3f} "
                  f"(rand_sigma={rand_sigma} × score std {scores.std():.3f})")
    else:
        perturbed = scores

    # Sort candidates by the chosen score mode.
    # np.lexsort: LAST key is the primary sort key. Earlier keys break ties.
    nnz = cache["nnz"].astype(np.int64)
    if score_mode == "density":
        # Primary: score desc; tiebreak: nnz asc.
        order = np.lexsort((nnz, -perturbed))
    elif score_mode == "small-first":
        # Primary: nnz asc; tiebreak: score desc. Maximizes K by packing
        # thin neurons first so the canvas isn't blocked by big ones.
        order = np.lexsort((-perturbed, nnz))
    elif score_mode == "hybrid":
        # Per-voxel score density: rewards small AND well-positioned neurons.
        util = perturbed / np.maximum(nnz.astype(np.float32), 1.0)
        order = np.lexsort((nnz, -util))
    else:
        raise ValueError(f"unknown score_mode: {score_mode!r}")
    if verbose:
        print(f"[select] score_mode = {score_mode}")
    if max_candidates_per_round is not None:
        order = order[:max_candidates_per_round]

    # C. Greedy fill.
    occupied = np.zeros(canvas_zyx, dtype=bool)
    selected: list[int] = []
    phase_by_idx: dict[int, str] = {}
    volumes: dict[int, tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

    # Shared in-memory read cache: avoids re-decoding the same .am during
    # the repair refill pass (which may iterate most of `order` again).
    read_cache: dict[int, tuple] = {}

    t2 = time.perf_counter()
    pbar = tqdm(total=N, desc="greedy fill", disable=not verbose)
    tried = 0
    gen = _prefetched_reads(order, cache, warp_dir, read_cache=read_cache)
    try:
        for idx, values, idx_tuple in gen:
            if len(selected) >= N:
                break
            tried += 1
            if values is None:
                continue
            zs, ys, xs = idx_tuple
            if occupied[zs, ys, xs].any():
                continue
            occupied[zs, ys, xs] = True
            selected.append(idx)
            phase_by_idx[idx] = "greedy"
            volumes[idx] = (values, (zs, ys, xs))
            pbar.update(1)
    finally:
        gen.close()
    pbar.close()
    if verbose:
        print(f"[select] greedy fill: got {len(selected)}/{N} in "
              f"{time.perf_counter()-t2:.1f}s (tried {tried} candidates)")

    # D. Verify C1 and repair. Maintain bbox counter incrementally to avoid
    #    rebuilding a 200+ MB int16 volume every round.
    counter = _init_bbox_counter(selected, cache, canvas_dims)
    remaining_order = [int(i) for i in order if int(i) not in set(selected)]
    for rnd in range(max_repair_rounds):
        ratios = _coverage_ratios(selected, counter, cache)
        violators_mask = ratios < coverage_threshold
        n_viol = int(violators_mask.sum())
        if verbose:
            print(f"[select] repair round {rnd}: C1 violators = {n_viol}/{len(selected)}, "
                  f"mean coverage = {ratios.mean():.3f}, min = {ratios.min():.3f}")
        if n_viol == 0:
            break

        # Drop violators: free voxels from occupied_mask and decrement counter.
        dropped = []
        for k, is_viol in enumerate(violators_mask):
            if is_viol:
                i = selected[k]
                _, (zs, ys, xs) = volumes[i]
                occupied[zs, ys, xs] = False
                _bbox_counter_remove(counter, i, cache)
                dropped.append(i)
        selected = [i for k, i in enumerate(selected) if not violators_mask[k]]
        for i in dropped:
            volumes.pop(i, None)

        # Refill from remaining candidates.
        refilled = 0
        need = N - len(selected)
        still_remaining = []
        t_refill = time.perf_counter()
        refill_tried = 0
        for idx in remaining_order:
            if refilled >= need:
                still_remaining.append(idx)
                continue
            refill_tried += 1
            try:
                values, (zs, ys, xs) = _load_neuron_canvas_voxels(idx, cache, warp_dir, read_cache)
            except Exception:
                continue
            if occupied[zs, ys, xs].any():
                still_remaining.append(idx)
                continue
            occupied[zs, ys, xs] = True
            selected.append(idx)
            phase_by_idx[idx] = "repair"
            volumes[idx] = (values, (zs, ys, xs))
            _bbox_counter_add(counter, idx, cache)
            refilled += 1
        remaining_order = still_remaining
        if verbose:
            print(f"[select]   dropped {len(dropped)}, refilled {refilled}, "
                  f"now {len(selected)}/{N}  "
                  f"(refill tried {refill_tried} in {time.perf_counter()-t_refill:.1f}s)")
        if refilled == 0:
            break

    # E. Expand pass — voxel-only, no C1 check. Squeeze in any remaining
    #    candidate that fits into the empty pockets left after repair. These
    #    are bonus neurons; they boost packing density and training
    #    richness but do NOT contribute to C1 satisfaction.
    if expand_after_repair and len(selected) < N:
        t_exp = time.perf_counter()
        expand_added = 0
        expand_tried = 0
        # Rebuild candidate pool from scratch using the CURRENT selected set
        # so dropped-then-not-refilled violators also get another chance here.
        # `remaining_order` alone would miss them because the repair loop only
        # tracked "never-in-selected" items.
        selected_set = set(selected)
        expand_order = [int(i) for i in order if int(i) not in selected_set]
        gen2 = _prefetched_reads(expand_order, cache, warp_dir, read_cache=read_cache)
        try:
            for idx, values, idx_tuple in gen2:
                if len(selected) >= N:
                    break
                expand_tried += 1
                if values is None:
                    continue
                zs, ys, xs = idx_tuple
                if occupied[zs, ys, xs].any():
                    continue
                occupied[zs, ys, xs] = True
                selected.append(idx)
                phase_by_idx[idx] = "expand"
                volumes[idx] = (values, (zs, ys, xs))
                _bbox_counter_add(counter, idx, cache)
                expand_added += 1
        finally:
            gen2.close()
        if verbose:
            print(f"[select] expand pass: +{expand_added} (tried {expand_tried}) in "
                  f"{time.perf_counter()-t_exp:.1f}s, K = {len(selected)}")

    # Final C1 check on the settled set.
    ratios = _coverage_ratios(selected, counter, cache)

    vol_list = [(i, volumes[i][0], volumes[i][1]) for i in selected]
    phases = np.array([phase_by_idx[i] for i in selected], dtype=object)
    return SelectionResult(
        indices=np.array(selected, dtype=np.int64),
        bbox_coverages=ratios,
        phases=phases,
        occupied_mask=occupied,
        volumes=vol_list,
        seed_used=int(seed),
    )
