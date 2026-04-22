"""Gaussian noise for intensity volumes.

Training realistic single-neuron segmentation models wants intensity
volumes that aren't pristine — real confocal stacks carry shot noise,
autofluorescence, camera readout variance, etc. Adding additive
Gaussian noise to `intensity.am` (but NOT `labels.am`) is the cheapest
way to inject that texture: the label ground-truth stays perfectly
crisp, but the input the model sees looks more like a microscope image.

Model: intensity' = intensity + baseline + N(0, sigma), clipped to
[0, 65535]. With baseline > 0 the background lands on a proper Gaussian
with mean = baseline and stdev = sigma, rather than a folded / half-
Gaussian you get from clipping at 0 without an offset. That matches real
sCMOS / CCD cameras which always report a positive dark-current offset
(typically ~100 DN) on top of the shot noise.

Defaults:
  * sigma = 50  — moderate; visible in the background, small compared to
    the 0..4095 peak intensities of FlyCircuit warps.
  * baseline = 100  — puts the background mean at 100 (~2 sigma above 0),
    so clipping at 0 effectively never triggers and the noise stays
    Gaussian.

Memory: generating a float32 noise volume temporarily doubles peak
memory (an extra ~820 MB for the default 989×646×337 canvas). Small
price for the simpler code.
"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    intensity: np.ndarray,
    sigma: float,
    baseline: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Add `baseline + N(0, sigma)` to a uint16 intensity volume, clip to [0, 65535].

    Returns a new uint16 array; `intensity` is not modified in place.
    When sigma == 0 and baseline == 0 the input is returned untouched
    (no RNG call, no allocation), so callers can pass both to disable noise.
    """
    if intensity.dtype != np.uint16:
        raise ValueError(f"intensity must be uint16; got {intensity.dtype}")
    if sigma <= 0 and baseline == 0:
        return intensity

    rng = np.random.default_rng(seed)
    # Noise in float32 to avoid rounding during accumulation; then clip and
    # cast back. Doing the add in uint16 directly would wrap negatives.
    if sigma > 0:
        noise = rng.normal(loc=0.0, scale=float(sigma),
                           size=intensity.shape).astype(np.float32)
    else:
        noise = np.zeros(intensity.shape, dtype=np.float32)
    noisy = intensity.astype(np.float32, copy=False) + float(baseline) + noise
    np.clip(noisy, 0.0, 65535.0, out=noisy)
    return noisy.astype(np.uint16)
