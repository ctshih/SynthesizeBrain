"""Gaussian noise for intensity volumes.

Training realistic single-neuron segmentation models wants intensity
volumes that aren't pristine — real confocal stacks carry shot noise,
autofluorescence, camera readout variance, etc. Adding additive
Gaussian noise to `intensity.am` (but NOT `labels.am`) is the cheapest
way to inject that texture: the label ground-truth stays perfectly
crisp, but the input the model sees looks more like a microscope image.

Spec:
  * Gaussian N(0, sigma), applied per-voxel independently
  * Clipped to [0, 65535] so the uint16 domain is preserved and
    negative values from the noise tail become 0 (physically what a
    camera sensor with a zero-offset would record)
  * Same noise volume is added to background voxels AND to neuron
    voxels — the background then has realistic shot-noise texture
    instead of being exactly 0, which matches real microscopy
  * Deterministic given a seed, so rerunning with the same `--seed`
    and `--noise-sigma` produces identical output

Memory: generating a float32 noise volume temporarily doubles peak
memory (an extra ~820 MB for the default 989×646×337 canvas). Small
price for the simpler code.
"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    intensity: np.ndarray,
    sigma: float,
    seed: int | None = None,
) -> np.ndarray:
    """Add N(0, sigma) noise to a uint16 intensity volume, clip to [0, 65535].

    Returns a new uint16 array; `intensity` is not modified in place.
    When sigma == 0 the input is returned untouched (no RNG call, no
    allocation), so callers can pass sigma=0 to disable noise.
    """
    if intensity.dtype != np.uint16:
        raise ValueError(f"intensity must be uint16; got {intensity.dtype}")
    if sigma <= 0:
        return intensity

    rng = np.random.default_rng(seed)
    # Noise in float32 to avoid rounding during accumulation; then clip and
    # cast back. Doing the add in uint16 directly would wrap negatives.
    noise = rng.normal(loc=0.0, scale=float(sigma),
                       size=intensity.shape).astype(np.float32)
    noisy = intensity.astype(np.float32, copy=False) + noise
    np.clip(noisy, 0.0, 65535.0, out=noisy)
    return noisy.astype(np.uint16)
