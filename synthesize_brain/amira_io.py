"""AmiraMesh BINARY-LITTLE-ENDIAN reader/writer for ushort lattices.

Reader adapted from the Kaleido project
(`C:\\Users\\USER\\Work\\Kaleido\\kaleido\\amira_io.py`). Only the `ushort`
code path is needed here — the FlyCircuit warp files are uniform-coordinate
single-lattice `ushort` volumes, and our synthesis outputs the same dtype
(ushort for intensity, uint16 for instance labels; the on-disk encoding is
identical).

Output uses raw (uncompressed) binary — simpler, and Amira opens it just fine.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class AmiraVolume:
    """A uniform 3D lattice loaded from an AmiraMesh file."""

    data: np.ndarray  # shape (Z, Y, X), dtype uint16
    dims: tuple[int, int, int]  # (X, Y, Z) as declared in header
    bbox: tuple[float, float, float, float, float, float]  # xmin xmax ymin ymax zmin zmax
    dtype_name: str

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        xmin, xmax, ymin, ymax, zmin, zmax = self.bbox
        nx, ny, nz = self.dims
        dx = (xmax - xmin) / max(nx - 1, 1)
        dy = (ymax - ymin) / max(ny - 1, 1)
        dz = (zmax - zmin) / max(nz - 1, 1)
        return dx, dy, dz

    @property
    def origin(self) -> tuple[float, float, float]:
        return self.bbox[0], self.bbox[2], self.bbox[4]


def _split_header_and_payload(raw: bytes) -> tuple[str, bytes]:
    marker = b"# Data section follows"
    idx = raw.find(marker)
    if idx < 0:
        raise ValueError("Not a valid AmiraMesh file: data section marker missing")
    nl1 = raw.find(b"\n", idx)
    nl2 = raw.find(b"\n", nl1 + 1)
    if nl1 < 0 or nl2 < 0:
        raise ValueError("Malformed AmiraMesh data section header")
    header = raw[:nl2 + 1].decode("latin-1", errors="replace")
    payload = raw[nl2 + 1:]
    return header, payload


_DTYPE_MAP = {
    "byte": ("<u1", 1),
    "ubyte": ("<u1", 1),
    "ushort": ("<u2", 2),
    "short": ("<i2", 2),
    "uint": ("<u4", 4),
    "int": ("<i4", 4),
    "float": ("<f4", 4),
}


def read_amira(path: str | Path) -> AmiraVolume:
    """Read an AmiraMesh BINARY-LITTLE-ENDIAN ushort volume (HxZip or raw)."""
    path = Path(path)
    raw = path.read_bytes()
    header, payload = _split_header_and_payload(raw)

    m = re.search(r"define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)", header)
    if not m:
        raise ValueError(f"Cannot find 'define Lattice' in {path}")
    nx, ny, nz = int(m.group(1)), int(m.group(2)), int(m.group(3))

    m = re.search(
        r"BoundingBox\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        header,
    )
    if not m:
        raise ValueError(f"Cannot find BoundingBox in {path}")
    bbox = tuple(float(m.group(i)) for i in range(1, 7))

    m = re.search(
        r"Lattice\s*\{\s*(\w+)\s+Data\s*\}\s*@(\d+)(?:\(([A-Za-z]+)\s*,\s*(\d+)\))?",
        header,
    )
    if not m:
        raise ValueError(f"Cannot parse Lattice descriptor in {path}")
    dtype_name = m.group(1)
    encoding = m.group(3)
    encoded_size = int(m.group(4)) if m.group(4) else None

    if dtype_name not in _DTYPE_MAP:
        raise NotImplementedError(f"Unsupported dtype: {dtype_name}")
    np_dtype, elem_size = _DTYPE_MAP[dtype_name]

    expected_elems = nx * ny * nz
    expected_bytes = expected_elems * elem_size

    if encoding == "HxZip":
        if encoded_size is None:
            raise ValueError("HxZip declared without compressed size")
        buf = zlib.decompress(payload[:encoded_size])
    elif encoding is None:
        buf = payload[:expected_bytes]
    else:
        raise NotImplementedError(f"Encoding {encoding} not supported")

    if len(buf) != expected_bytes:
        raise ValueError(
            f"Decoded byte count {len(buf)} != expected {expected_bytes} for {path}"
        )

    # AmiraMesh stores X-fastest then Y then Z, so reshape (Z, Y, X).
    arr = np.frombuffer(buf, dtype=np_dtype).reshape((nz, ny, nx))
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    return AmiraVolume(data=arr, dims=(nx, ny, nz), bbox=bbox, dtype_name=dtype_name)


def parse_header(path: str | Path) -> tuple[tuple[int, int, int], tuple[float, ...]]:
    """Read only the ASCII header (no zlib decode) and return (dims, bbox).

    Cheap — lets us enumerate all 9987 files without decompressing any payload.
    """
    path = Path(path)
    with open(path, "rb") as fh:
        # Headers are < 2 KB; a small read is enough.
        head_bytes = fh.read(4096)
    marker = b"# Data section follows"
    if marker not in head_bytes:
        # Fall back to reading more if the header is unusually long.
        head_bytes = path.read_bytes()[:16384]
    header = head_bytes.decode("latin-1", errors="replace")

    m = re.search(r"define\s+Lattice\s+(\d+)\s+(\d+)\s+(\d+)", header)
    if not m:
        raise ValueError(f"Cannot find 'define Lattice' in {path}")
    dims = (int(m.group(1)), int(m.group(2)), int(m.group(3)))

    m = re.search(
        r"BoundingBox\s+(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)\s+"
        r"(-?\d+\.?\d*(?:[eE][-+]?\d+)?)",
        header,
    )
    if not m:
        raise ValueError(f"Cannot find BoundingBox in {path}")
    bbox = tuple(float(m.group(i)) for i in range(1, 7))
    return dims, bbox


def write_ushort_amira(
    path: str | Path,
    data: np.ndarray,
    bbox: tuple[float, float, float, float, float, float],
) -> None:
    """Write a uint16 volume as AmiraMesh BINARY-LITTLE-ENDIAN, raw (uncompressed).

    `data` shape: (Z, Y, X), dtype uint16.
    """
    if data.ndim != 3:
        raise ValueError(f"data must be 3D (Z, Y, X); got shape {data.shape}")
    if data.dtype != np.uint16:
        raise ValueError(f"data must be uint16; got {data.dtype}")

    nz, ny, nx = data.shape
    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    header = (
        "# AmiraMesh BINARY-LITTLE-ENDIAN 2.1\n"
        "\n\n"
        f"define Lattice {nx} {ny} {nz}\n"
        "\n"
        "Parameters {\n"
        f'    Content "{nx}x{ny}x{nz} ushort, uniform coordinates",\n'
        f"    BoundingBox {xmin} {xmax} {ymin} {ymax} {zmin} {zmax},\n"
        '    CoordType "uniform"\n'
        "}\n"
        "\n"
        "Lattice { ushort Data } @1\n"
        "\n"
        "# Data section follows\n"
        "@1\n"
    )

    # AmiraMesh expects X-fastest, Y, Z; (Z, Y, X) C-contiguous already matches.
    flat = np.ascontiguousarray(data).tobytes()

    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(flat)
