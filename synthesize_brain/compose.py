"""Phase 3: paste selected neurons into paired intensity + label volumes."""

from __future__ import annotations

import numpy as np

from .select import SelectionResult


def compose(
    sel: SelectionResult,
    canvas_dims: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (intensity_zyx_u16, labels_zyx_u16) from the selection.

    Label IDs run 1..K where K = len(sel.indices); background = 0.
    Ordering of labels follows the order neurons were accepted during
    selection.
    """
    nx_c, ny_c, nz_c = canvas_dims
    shape_zyx = (nz_c, ny_c, nx_c)
    intensity = np.zeros(shape_zyx, dtype=np.uint16)
    labels = np.zeros(shape_zyx, dtype=np.uint16)

    K = len(sel.volumes)
    if K > np.iinfo(np.uint16).max:
        raise ValueError(f"K={K} exceeds uint16 label capacity")

    for label_id, (_cache_idx, values, (zs, ys, xs)) in enumerate(sel.volumes, start=1):
        intensity[zs, ys, xs] = values
        labels[zs, ys, xs] = label_id

    return intensity, labels
