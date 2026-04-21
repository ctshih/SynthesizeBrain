"""Three-axis maximum-intensity-projection previews for quick eyeballing."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _random_label_colors(num_labels: int, seed: int = 0) -> np.ndarray:
    """Map label IDs 0..num_labels to RGB uint8; 0 = black."""
    rng = np.random.default_rng(seed)
    colors = rng.integers(64, 256, size=(num_labels + 1, 3), dtype=np.uint8)
    colors[0] = 0
    return colors


def write_mip(
    path: str | Path,
    intensity_zyx: np.ndarray,
    labels_zyx: np.ndarray,
    seed: int = 0,
) -> None:
    """Save a 2x3 grid PNG: row 1 = intensity MIPs (grey), row 2 = label MIPs (random colors)."""
    # Intensity MIPs along each axis.
    int_mip_z = intensity_zyx.max(axis=0)  # (Y, X)
    int_mip_y = intensity_zyx.max(axis=1)  # (Z, X)
    int_mip_x = intensity_zyx.max(axis=2)  # (Z, Y)

    # For labels we want the label with the strongest intensity at each column — but
    # since labels are disjoint (constraint 2), argmax along an axis picks the one
    # label that has any voxel along that column. Easier: take the max label along
    # the axis (since labels > 0 where present). Any nonzero voxel contributes.
    lbl_mip_z = labels_zyx.max(axis=0)
    lbl_mip_y = labels_zyx.max(axis=1)
    lbl_mip_x = labels_zyx.max(axis=2)

    K = int(labels_zyx.max())
    colors = _random_label_colors(K, seed=seed)
    lbl_mip_z_rgb = colors[lbl_mip_z]
    lbl_mip_y_rgb = colors[lbl_mip_y]
    lbl_mip_x_rgb = colors[lbl_mip_x]

    # Normalize intensity for display.
    def _norm(img: np.ndarray) -> np.ndarray:
        m = float(img.max())
        if m <= 0:
            return img.astype(np.float32)
        return (img.astype(np.float32) / m).clip(0, 1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    titles = ["MIP along Z (axial)", "MIP along Y (coronal)", "MIP along X (sagittal)"]
    for col, (int_img, lbl_img, title) in enumerate(
        zip([int_mip_z, int_mip_y, int_mip_x],
            [lbl_mip_z_rgb, lbl_mip_y_rgb, lbl_mip_x_rgb],
            titles)
    ):
        axes[0, col].imshow(_norm(int_img), cmap="gray", origin="lower")
        axes[0, col].set_title(f"intensity — {title}")
        axes[0, col].axis("off")
        axes[1, col].imshow(lbl_img, origin="lower")
        axes[1, col].set_title(f"labels — {title}")
        axes[1, col].axis("off")

    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
