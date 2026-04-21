"""Phase 1: scan every warp .am file, extract per-neuron metadata, and cache.

For each file we record:
- filename, driver (GAL4 driver prefix: Tdc2 / Trh / VGlut / fru)
- lattice dims (nx, ny, nz) of the cropped volume
- physical bbox of the cropped volume (from header)
- tight physical bbox — the sub-bbox that actually contains non-zero voxels
- non-zero voxel count
- max intensity

After all files are scanned we derive the canvas (union of all header bboxes,
rounded to integer voxel grid) and precompute each neuron's voxel-space origin
and tight bbox on that canvas.

The cache is an .npz keyed by neuron index (0..N-1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .amira_io import read_amira


@dataclass
class NeuronRecord:
    filename: str
    driver: str
    dims: tuple[int, int, int]            # lattice nx, ny, nz
    bbox: tuple[float, ...]               # cropped lattice bbox (physical), 6 floats
    tight_bbox: tuple[float, ...]         # non-zero support bbox (physical), 6 floats, inclusive
    nnz: int
    max_intensity: int


def _driver_from_filename(name: str) -> str:
    # e.g. "Tdc2-F-000000_seg001_warp_volume.am" -> "Tdc2"
    return name.split("-", 1)[0]


def _scan_one(path_str: str) -> NeuronRecord | None:
    """Worker: load one .am, compute tight bbox + nnz. Returns None on failure."""
    try:
        path = Path(path_str)
        vol = read_amira(path)
        data = vol.data  # (Z, Y, X) uint16
        nz_mask = data > 0
        nnz = int(nz_mask.sum())
        if nnz == 0:
            return None  # skip empty volumes

        # Tight bbox in voxel indices (local lattice frame, inclusive).
        z_any = nz_mask.any(axis=(1, 2))
        y_any = nz_mask.any(axis=(0, 2))
        x_any = nz_mask.any(axis=(0, 1))
        zmin_v, zmax_v = int(z_any.argmax()), int(len(z_any) - 1 - z_any[::-1].argmax())
        ymin_v, ymax_v = int(y_any.argmax()), int(len(y_any) - 1 - y_any[::-1].argmax())
        xmin_v, xmax_v = int(x_any.argmax()), int(len(x_any) - 1 - x_any[::-1].argmax())

        # Convert local voxel indices -> physical coords using the file's bbox.
        dx, dy, dz = vol.voxel_size
        ox, oy, oz = vol.origin
        tight_bbox = (
            ox + xmin_v * dx,
            ox + xmax_v * dx,
            oy + ymin_v * dy,
            oy + ymax_v * dy,
            oz + zmin_v * dz,
            oz + zmax_v * dz,
        )
        return NeuronRecord(
            filename=path.name,
            driver=_driver_from_filename(path.name),
            dims=vol.dims,
            bbox=vol.bbox,
            tight_bbox=tight_bbox,
            nnz=nnz,
            max_intensity=int(data.max()),
        )
    except Exception as e:
        print(f"[index] skip {path_str}: {e}")
        return None


def _pack_canvas(records: list[NeuronRecord]) -> tuple[tuple[float, ...], tuple[int, int, int], tuple[float, float, float]]:
    """Return (canvas_bbox, canvas_dims, voxel_size) spanning every record."""
    # Derive voxel spacing from the first record (should be ~1 for all FlyCircuit warps).
    r0 = records[0]
    nx0, ny0, nz0 = r0.dims
    bx0, bx1, by0, by1, bz0, bz1 = r0.bbox
    vx = (bx1 - bx0) / max(nx0 - 1, 1)
    vy = (by1 - by0) / max(ny0 - 1, 1)
    vz = (bz1 - bz0) / max(nz0 - 1, 1)

    xmin = min(r.bbox[0] for r in records)
    xmax = max(r.bbox[1] for r in records)
    ymin = min(r.bbox[2] for r in records)
    ymax = max(r.bbox[3] for r in records)
    zmin = min(r.bbox[4] for r in records)
    zmax = max(r.bbox[5] for r in records)

    nx = int(round((xmax - xmin) / vx)) + 1
    ny = int(round((ymax - ymin) / vy)) + 1
    nz = int(round((zmax - zmin) / vz)) + 1
    return (xmin, xmax, ymin, ymax, zmin, zmax), (nx, ny, nz), (vx, vy, vz)


def build_index(
    warp_dir: str | Path,
    cache_path: str | Path,
    n_workers: int | None = None,
    limit: int | None = None,
) -> dict:
    """Scan all *.am files in `warp_dir`, build the index, and save to `cache_path`.

    Returns the index dict (also saved to .npz).
    """
    warp_dir = Path(warp_dir)
    files = sorted(warp_dir.glob("*.am"))
    if limit is not None:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"No .am files found in {warp_dir}")

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    print(f"[index] scanning {len(files)} files with {n_workers} workers...")
    records: list[NeuronRecord] = []
    if n_workers == 1:
        for p in tqdm(files, desc="index"):
            r = _scan_one(str(p))
            if r is not None:
                records.append(r)
    else:
        with Pool(n_workers) as pool:
            for r in tqdm(
                pool.imap_unordered(_scan_one, [str(p) for p in files], chunksize=16),
                total=len(files),
                desc="index",
            ):
                if r is not None:
                    records.append(r)

    # Stable order by filename so runs are reproducible regardless of worker scheduling.
    records.sort(key=lambda r: r.filename)

    canvas_bbox, canvas_dims, voxel_size = _pack_canvas(records)
    vx, vy, vz = voxel_size
    cx0, _, cy0, _, cz0, _ = canvas_bbox

    N = len(records)
    filenames = np.array([r.filename for r in records], dtype=object)
    drivers = np.array([r.driver for r in records], dtype=object)
    dims_arr = np.array([r.dims for r in records], dtype=np.int32)
    bbox_arr = np.array([r.bbox for r in records], dtype=np.float64)
    tight_bbox_arr = np.array([r.tight_bbox for r in records], dtype=np.float64)
    nnz_arr = np.array([r.nnz for r in records], dtype=np.int64)
    max_int_arr = np.array([r.max_intensity for r in records], dtype=np.int32)

    # Canvas-space voxel coords for each neuron.
    # origin_ix/iy/iz: voxel coord in canvas where the cropped lattice begins.
    origin_ix = np.round((bbox_arr[:, 0] - cx0) / vx).astype(np.int32)
    origin_iy = np.round((bbox_arr[:, 2] - cy0) / vy).astype(np.int32)
    origin_iz = np.round((bbox_arr[:, 4] - cz0) / vz).astype(np.int32)

    # Tight bbox in canvas voxels (inclusive min, inclusive max).
    tight_ix_min = np.round((tight_bbox_arr[:, 0] - cx0) / vx).astype(np.int32)
    tight_ix_max = np.round((tight_bbox_arr[:, 1] - cx0) / vx).astype(np.int32)
    tight_iy_min = np.round((tight_bbox_arr[:, 2] - cy0) / vy).astype(np.int32)
    tight_iy_max = np.round((tight_bbox_arr[:, 3] - cy0) / vy).astype(np.int32)
    tight_iz_min = np.round((tight_bbox_arr[:, 4] - cz0) / vz).astype(np.int32)
    tight_iz_max = np.round((tight_bbox_arr[:, 5] - cz0) / vz).astype(np.int32)

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        filenames=filenames,
        drivers=drivers,
        dims=dims_arr,
        bbox=bbox_arr,
        tight_bbox=tight_bbox_arr,
        nnz=nnz_arr,
        max_intensity=max_int_arr,
        origin_ix=origin_ix,
        origin_iy=origin_iy,
        origin_iz=origin_iz,
        tight_ix_min=tight_ix_min,
        tight_ix_max=tight_ix_max,
        tight_iy_min=tight_iy_min,
        tight_iy_max=tight_iy_max,
        tight_iz_min=tight_iz_min,
        tight_iz_max=tight_iz_max,
        canvas_bbox=np.array(canvas_bbox, dtype=np.float64),
        canvas_dims=np.array(canvas_dims, dtype=np.int32),
        voxel_size=np.array(voxel_size, dtype=np.float64),
    )

    print(f"[index] N={N}, canvas dims {canvas_dims}, voxel size {voxel_size}")
    print(f"[index] cache written: {cache_path} ({cache_path.stat().st_size/1024:.1f} KiB)")

    return {
        "N": N,
        "canvas_bbox": canvas_bbox,
        "canvas_dims": canvas_dims,
        "voxel_size": voxel_size,
    }


def load_index(cache_path: str | Path) -> dict:
    """Load the Phase 1 cache as a dict of arrays."""
    cache_path = Path(cache_path)
    data = np.load(cache_path, allow_pickle=True)
    return {k: data[k] for k in data.files}
