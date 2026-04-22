"""Pairwise voxel-level contact statistics between selected neurons.

For every pair of distinct labels (a, b) we count three contact types based
on the 26-neighbourhood of each a-labelled voxel:

  F (Face)   — neighbour voxel shares a full face
               (|Δx|+|Δy|+|Δz| = 1, exactly one axis differs by 1)
  E (Edge)   — neighbour voxel shares only an edge
               (exactly two of |Δx|, |Δy|, |Δz| equal 1, third is 0)
  V (Vertex) — neighbour voxel shares only a corner
               (all three of |Δx|, |Δy|, |Δz| equal 1)

Because labels are voxel-disjoint by construction, we never double-count a
voxel overlap; each count is a genuine touch without overlap. Each
voxel-pair (v in a, w in b) is counted once even though each side of the
offset would visit it, by canonicalising to `a < b`.

Output CSV columns: `neuron1, neuron2, N_F, N_E, N_V` with `neuron1` /
`neuron2` being the source `.am` filenames in ascending order. Only pairs
with at least one contact voxel are emitted, since K is ~150 and the
K × (K − 1) / 2 full table would be mostly zeros.
"""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np


# 26-neighbour offsets classified by number of non-zero coordinates.
# In (Δz, Δy, Δx) order to match the (Z, Y, X) layout of `labels`.
_OFFSETS: list[tuple[int, int, int, str]] = []
for _dz, _dy, _dx in itertools.product((-1, 0, 1), repeat=3):
    _n = (_dz != 0) + (_dy != 0) + (_dx != 0)
    if _n == 0:
        continue
    _t = "F" if _n == 1 else ("E" if _n == 2 else "V")
    _OFFSETS.append((_dz, _dy, _dx, _t))
# 6 F + 12 E + 8 V = 26 offsets.


def _slice_pair(d: int, size: int) -> tuple[slice, slice]:
    """Return the (dst_slice, src_slice) such that `arr[dst] == arr_shifted[src]`
    when arr_shifted is arr shifted by +d along this axis. Shared length is
    size - |d|, always positive when |d| < size."""
    if d > 0:
        return slice(0, size - d), slice(d, size)
    elif d < 0:
        return slice(-d, size), slice(0, size + d)
    else:
        return slice(0, size), slice(0, size)


def compute_contact_counts(labels: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Return a dict mapping (a, b) with a < b → [N_F, N_E, N_V].

    `labels` is (Z, Y, X) uint16 with 0 = background, 1..K = neuron IDs.
    """
    if labels.ndim != 3:
        raise ValueError(f"expected 3D labels, got {labels.shape}")
    Z, Y, X = labels.shape
    counts: dict[tuple[int, int], list[int]] = {}

    for dz, dy, dx, t in _OFFSETS:
        zd, zs = _slice_pair(dz, Z)
        yd, ys = _slice_pair(dy, Y)
        xd, xs = _slice_pair(dx, X)
        a = labels[zd, yd, xd]  # "home" voxel labels
        b = labels[zs, ys, xs]  # neighbour labels (shifted by (dz,dy,dx))
        # Keep voxel pairs that are both non-zero, non-equal, and canonicalise
        # to a < b so each unordered pair is counted exactly once across the
        # 26 offsets (since offset and its negative visit the same pair with
        # (a,b) swapped).
        mask = (a != 0) & (b != 0) & (a < b)
        if not mask.any():
            continue
        a_m = a[mask].astype(np.int64)
        b_m = b[mask].astype(np.int64)
        # Pack (a, b) into a single int key for fast histogramming.
        # Enough bits: max label ~65535, so a * (1<<17) + b fits in int64.
        key = (a_m << 17) | b_m
        uniq, cnt = np.unique(key, return_counts=True)
        tidx = {"F": 0, "E": 1, "V": 2}[t]
        for k, c in zip(uniq, cnt):
            a_val = int(k >> 17)
            b_val = int(k & ((1 << 17) - 1))
            row = counts.setdefault((a_val, b_val), [0, 0, 0])
            row[tidx] += int(c)

    return counts


def write_contacts_csv(
    path: str | Path,
    labels: np.ndarray,
    label_to_name: dict[int, str],
) -> int:
    """Compute contact counts and write CSV. Returns number of rows written."""
    counts = compute_contact_counts(labels)
    # Sort by pair for reproducible diffs.
    rows = sorted(counts.items())
    path = Path(path)
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["neuron1", "neuron2", "N_F", "N_E", "N_V"])
        for (a, b), (nf, ne, nv) in rows:
            w.writerow([label_to_name[a], label_to_name[b], nf, ne, nv])
    return len(rows)
