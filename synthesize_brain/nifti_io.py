"""NIfTI writers for the synthesized volumes.

We emit NIfTI-1 (.nii.gz). Voxel spacing is carried in the affine; origin is
set to the canvas bbox min corner so that physical coordinates of a voxel in
the output match those in the source warp files. The affine is purely
diagonal (axis-aligned), which matches the uniform-coordinate assumption of
the input AmiraMesh files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib


def write_nifti(
    path: str | Path,
    data_zyx: np.ndarray,
    canvas_bbox: tuple[float, float, float, float, float, float],
    voxel_size: tuple[float, float, float],
) -> None:
    """Write a (Z, Y, X) array as .nii.gz.

    Nibabel stores data in (X, Y, Z) order, so we transpose on the way in.
    """
    if data_zyx.ndim != 3:
        raise ValueError(f"expected 3D (Z, Y, X); got {data_zyx.shape}")
    data_xyz = np.ascontiguousarray(np.transpose(data_zyx, (2, 1, 0)))

    vx, vy, vz = voxel_size
    xmin, _, ymin, _, zmin, _ = canvas_bbox
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0] = vx
    affine[1, 1] = vy
    affine[2, 2] = vz
    affine[0, 3] = xmin
    affine[1, 3] = ymin
    affine[2, 3] = zmin

    img = nib.Nifti1Image(data_xyz, affine)
    nib.save(img, str(path))
