"""Scan-video generator: one frame per label, newest bright, seen dim.

Produces an MP4 that walks through the K neurons in the label volume in
label-ID order. On frame i:
  - voxels with label == i are rendered full white (the "current" neuron)
  - voxels with label < i are rendered dim grey (the "already seen" pile)
  - everything else is black
  - the top-left corner gets a "N=i" caption

This makes a quick per-label confidence check: each neuron pops out in
white once, then fades into the grey backdrop as the cursor moves on.

Rendering is MIP-based and incremental. For each of the 3 canonical axes
we carry a cumulative uint8 grayscale image of "voxels that have been
rendered as current at some earlier frame"; drawing frame i then just
places the i-th label's footprint at full brightness on top. That
keeps the per-frame cost to O(size of label i), not O(canvas).
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# Colour spec: current label white, seen pile a dim grey, background black.
_BRIGHT = 255
_DIM = 80
_TEXT_COLOR = (255, 255, 0)     # yellow — pops on both black and dim grey
_TEXT_SIZE_PX = 48              # roughly proportional to a 1400-px-wide frame


def _label_footprints(labels: np.ndarray, K: int):
    """For each label 1..K, return three (row, col) tuples — one per axis —
    that mark the 2D MIP footprint of that label along that axis.

    Returned: list `[(ys_axial, xs_axial, zs_coronal, xs_coronal, zs_sagittal, ys_sagittal), ...]`
    indexed 0..K-1 for labels 1..K.
    """
    Z, Y, X = labels.shape
    out = []
    for i in range(1, K + 1):
        zs, ys, xs = np.where(labels == i)
        # Dedupe per-axis MIP pixel sets so we don't re-write the same pixel.
        # np.unique on row-major pair keeps memory bounded.
        def _unique_pair(a, b):
            key = a.astype(np.int64) * (1 << 20) + b.astype(np.int64)
            uniq = np.unique(key)
            return (uniq >> 20).astype(np.int32), (uniq & ((1 << 20) - 1)).astype(np.int32)

        y_ax, x_ax = _unique_pair(ys, xs)     # MIP along Z -> image (Y, X)
        z_co, x_co = _unique_pair(zs, xs)     # MIP along Y -> image (Z, X)
        z_sa, y_sa = _unique_pair(zs, ys)     # MIP along X -> image (Z, Y)
        out.append((y_ax, x_ax, z_co, x_co, z_sa, y_sa))
    return out


def _load_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try a few common TTF paths; fall back to Pillow's bitmap default."""
    candidates = [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size_px)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def write_scan_video(
    labels: np.ndarray,
    output_path: str | Path,
    fps: float = 3.0,
    dim_brightness: int = _DIM,
    highlight_brightness: int = _BRIGHT,
    verbose: bool = True,
) -> None:
    """Generate `scan_video.mp4` from a (Z, Y, X) uint16/uint8 label volume.

    K frames (one per non-background label), laid out as three MIPs side-
    by-side (axial | coronal | sagittal), with a "N={i}" caption in the
    top-left corner of the axial view.
    """
    if labels.ndim != 3:
        raise ValueError(f"labels must be 3D; got {labels.shape}")
    Z, Y, X = labels.shape
    K = int(labels.max())
    if K < 1:
        raise ValueError("labels has no non-zero values — nothing to animate")

    footprints = _label_footprints(labels, K)

    # MIP canvas sizes per axis.
    H_ax, W_ax = Y, X         # axial:   MIP along Z
    H_co, W_co = Z, X         # coronal: MIP along Y
    H_sa, W_sa = Z, Y         # sagittal:MIP along X
    gap = 8                   # pixel separator between panels
    H_frame = max(H_ax, H_co, H_sa)
    W_frame = W_ax + gap + W_co + gap + W_sa

    # Cumulative "seen" layers (dim grey where any earlier label landed).
    seen_ax = np.zeros((H_ax, W_ax), dtype=np.uint8)
    seen_co = np.zeros((H_co, W_co), dtype=np.uint8)
    seen_sa = np.zeros((H_sa, W_sa), dtype=np.uint8)

    font = _load_font(_TEXT_SIZE_PX)

    path = Path(output_path)
    # macro_block_size=1 so tiny dimensions don't get padded; pixelformat yuv420p
    # so Windows Media Player / QuickTime can play the resulting mp4.
    writer = imageio.get_writer(
        str(path),
        fps=fps,
        codec="libx264",
        quality=7,
        macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )

    try:
        it = tqdm(range(1, K + 1), desc="scan video", disable=not verbose)
        for i in it:
            y_ax, x_ax, z_co, x_co, z_sa, y_sa = footprints[i - 1]

            # Draw "current" as a fresh bright mask, then composite max(seen, bright).
            cur_ax = np.zeros_like(seen_ax)
            cur_ax[y_ax, x_ax] = highlight_brightness
            cur_co = np.zeros_like(seen_co)
            cur_co[z_co, x_co] = highlight_brightness
            cur_sa = np.zeros_like(seen_sa)
            cur_sa[z_sa, y_sa] = highlight_brightness

            view_ax = np.maximum(seen_ax, cur_ax)
            view_co = np.maximum(seen_co, cur_co)
            view_sa = np.maximum(seen_sa, cur_sa)

            # Assemble the stitched frame. Pad shorter views to frame height.
            frame = np.zeros((H_frame, W_frame), dtype=np.uint8)
            # Bottom-align each panel so everything sits on the same floor.
            frame[H_frame - H_ax:, :W_ax] = view_ax
            frame[H_frame - H_co:, W_ax + gap:W_ax + gap + W_co] = view_co
            frame[H_frame - H_sa:, W_ax + gap + W_co + gap:] = view_sa

            # RGB for video + text overlay.
            rgb = np.stack([frame, frame, frame], axis=-1)
            img = Image.fromarray(rgb)
            draw = ImageDraw.Draw(img)
            draw.text((16, 12), f"N = {i}", fill=_TEXT_COLOR, font=font)

            # Also caption which panel is which, once, along the bottom of frame 1.
            if i == 1:
                small = _load_font(max(_TEXT_SIZE_PX // 2, 14))
                draw.text((16, H_frame - 32), "axial", fill=(180, 180, 180), font=small)
                draw.text((W_ax + gap + 16, H_frame - 32), "coronal",
                          fill=(180, 180, 180), font=small)
                draw.text((W_ax + gap + W_co + gap + 16, H_frame - 32), "sagittal",
                          fill=(180, 180, 180), font=small)

            writer.append_data(np.array(img))

            # Promote this label from "current bright" to "already seen dim"
            # so subsequent frames show it in the dim pile.
            seen_ax[y_ax, x_ax] = np.maximum(seen_ax[y_ax, x_ax], dim_brightness)
            seen_co[z_co, x_co] = np.maximum(seen_co[z_co, x_co], dim_brightness)
            seen_sa[z_sa, y_sa] = np.maximum(seen_sa[z_sa, y_sa], dim_brightness)
    finally:
        writer.close()

    if verbose:
        print(f"[scan_video] wrote {K} frames to {path} "
              f"({path.stat().st_size / 1024:.0f} KiB)")
