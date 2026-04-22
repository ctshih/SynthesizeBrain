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

    # Accept either `{ dtype Data }` (scalar / intensity) or `{ dtype Labels }`
    # (label field). The field-name slot captures but we don't use its value.
    m = re.search(
        r"Lattice\s*\{\s*(\w+)\s+(\w+)\s*\}\s*@(\d+)(?:\(([A-Za-z]+)\s*,\s*(\d+)\))?",
        header,
    )
    if not m:
        raise ValueError(f"Cannot parse Lattice descriptor in {path}")
    dtype_name = m.group(1)
    encoding = m.group(4)
    encoded_size = int(m.group(5)) if m.group(5) else None

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


def write_label_amira(
    path: str | Path,
    data: np.ndarray,
    bbox: tuple[float, float, float, float, float, float],
    label_colors: np.ndarray,
    label_names: list[str] | None = None,
) -> None:
    """Write a uint16 label volume as a real Avizo label field.

    Adds a `Materials { ... }` block and uses `Lattice { ushort Labels } @1`
    (not `Data`). Avizo then loads the file as HxUniformLabelField3, which:
      - uses nearest-neighbour sampling by default (no trilinear fade between
        adjacent labels when a bandpass colormap is applied);
      - shows a Materials list in the Properties area with per-label
        visibility toggles and per-label default colors;
      - still supports bandpass inspection via MinMax [N-1, N+1].

    `data`: (Z, Y, X) uint16. Values 0..K, 0 = background / Exterior.
    `label_colors`: (K+1, 3) uint8. Index 0 is ignored (Exterior gets no
        Color line, per Avizo convention); indices 1..K are per-label RGB.
    `label_names`: optional list of K+1 material names. Defaults to
        ["Exterior", "Neuron_0001", ..., f"Neuron_{K:04d}"].
    """
    if data.ndim != 3:
        raise ValueError(f"data must be 3D (Z, Y, X); got shape {data.shape}")
    if data.dtype != np.uint16:
        raise ValueError(f"data must be uint16; got {data.dtype}")

    K = int(data.max())
    if label_colors.ndim != 2 or label_colors.shape[0] < K + 1 or label_colors.shape[1] != 3:
        raise ValueError(
            f"label_colors must be (>={K+1}, 3); got shape {label_colors.shape}"
        )
    if label_colors.dtype != np.uint8:
        raise ValueError(f"label_colors must be uint8; got {label_colors.dtype}")
    if label_names is None:
        label_names = ["Exterior"] + [f"Neuron_{i:04d}" for i in range(1, K + 1)]
    if len(label_names) < K + 1:
        raise ValueError(f"label_names length {len(label_names)} < K+1 = {K+1}")

    # Materials block.
    lines = ["    Materials {"]
    for i in range(K + 1):
        name = label_names[i]
        lines.append(f"        {name} {{")
        if i == 0:
            # Exterior: just Id, no Color (matches Avizo sample files).
            lines.append(f"            Id {i}")
        else:
            r, g, b = (float(c) / 255.0 for c in label_colors[i])
            lines.append(f"            Id {i},")
            lines.append(f"            Color {r:.6f} {g:.6f} {b:.6f}")
        lines.append("        }")
    lines.append("    }")
    materials_block = "\n".join(lines)

    nz, ny, nx = data.shape
    xmin, xmax, ymin, ymax, zmin, zmax = bbox

    header = (
        "# Avizo BINARY-LITTLE-ENDIAN 2.1\n"
        "\n\n"
        f"define Lattice {nx} {ny} {nz}\n"
        "\n"
        "Parameters {\n"
        f"{materials_block}\n"
        f'    Content "{nx}x{ny}x{nz} ushort, uniform coordinates",\n'
        f"    BoundingBox {xmin} {xmax} {ymin} {ymax} {zmin} {zmax},\n"
        '    CoordType "uniform"\n'
        "}\n"
        "\n"
        "Lattice { ushort Labels } @1\n"
        "\n"
        "# Data section follows\n"
        "@1\n"
    )

    flat = np.ascontiguousarray(data).tobytes()

    path = Path(path)
    with open(path, "wb") as fh:
        fh.write(header.encode("ascii"))
        fh.write(flat)
