"""Small TIFF utilities for this competition's GeoTIFF-like files.

This avoids requiring rasterio/tifffile in the local environment. It supports
the formats observed in EDA:
- deflate-compressed uint8 satellite TIFFs, possibly multi-channel
- uncompressed float32 GPM-IMERG target/template TIFFs
"""

from __future__ import annotations

import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TIFF_TYPE_INFO = {
    1: ("B", 1),
    2: ("c", 1),
    3: ("H", 2),
    4: ("I", 4),
    5: ("II", 8),
    11: ("f", 4),
    12: ("d", 8),
}

TIFF_TAG_NAMES = {
    256: "width",
    257: "height",
    258: "bits_per_sample",
    259: "compression",
    262: "photometric",
    273: "strip_offsets",
    277: "samples_per_pixel",
    278: "rows_per_strip",
    279: "strip_byte_counts",
    284: "planar_config",
    317: "predictor",
    339: "sample_format",
}


@dataclass(frozen=True)
class TiffMeta:
    width: int
    height: int
    samples_per_pixel: int
    bits_per_sample: tuple[int, ...]
    sample_format: tuple[int, ...]
    compression: int
    rows_per_strip: int
    strip_offsets: tuple[int, ...]
    strip_byte_counts: tuple[int, ...]
    dtype: str
    endian: str


def _raw_value(data: bytes, endian: str, typ: int, count: int, raw: bytes) -> Any:
    fmt, size = TIFF_TYPE_INFO[typ]
    total = count * size
    if total <= 4:
        buf = raw[:total]
    else:
        offset = struct.unpack(endian + "I", raw)[0]
        buf = data[offset : offset + total]

    if typ == 2:
        return buf.rstrip(b"\0").decode("utf-8", "replace")
    if typ == 5:
        values = []
        for i in range(count):
            numerator, denominator = struct.unpack(endian + "II", buf[i * 8 : i * 8 + 8])
            values.append(numerator / denominator if denominator else math.nan)
        return values[0] if count == 1 else tuple(values)

    values = struct.unpack(endian + (fmt * count), buf)
    return values[0] if count == 1 else tuple(values)


def _as_tuple(value: Any) -> tuple[int, ...]:
    if isinstance(value, tuple):
        return tuple(int(v) for v in value)
    return (int(value),)


def _dtype_from_tags(bits_per_sample: tuple[int, ...], sample_format: tuple[int, ...]) -> np.dtype:
    bits = bits_per_sample[0]
    fmt = sample_format[0] if sample_format else 1
    if fmt == 1 and bits == 8:
        return np.dtype("uint8")
    if fmt == 1 and bits == 16:
        return np.dtype("uint16")
    if fmt == 3 and bits == 32:
        return np.dtype("float32")
    if fmt == 3 and bits == 64:
        return np.dtype("float64")
    raise ValueError(f"Unsupported TIFF dtype sample_format={fmt}, bits={bits}")


def read_tiff_tags(path: Path) -> tuple[bytes, TiffMeta]:
    data = path.read_bytes()
    if data[:2] == b"II":
        endian = "<"
    elif data[:2] == b"MM":
        endian = ">"
    else:
        raise ValueError(f"Not a TIFF file: {path}")

    magic = struct.unpack(endian + "H", data[2:4])[0]
    if magic != 42:
        raise ValueError(f"Unsupported TIFF magic {magic}: {path}")

    ifd_offset = struct.unpack(endian + "I", data[4:8])[0]
    tag_count = struct.unpack(endian + "H", data[ifd_offset : ifd_offset + 2])[0]
    tags: dict[str, Any] = {}
    for i in range(tag_count):
        offset = ifd_offset + 2 + i * 12
        tag, typ, count = struct.unpack(endian + "HHI", data[offset : offset + 8])
        raw = data[offset + 8 : offset + 12]
        name = TIFF_TAG_NAMES.get(tag, f"tag_{tag}")
        if typ in TIFF_TYPE_INFO:
            tags[name] = _raw_value(data, endian, typ, count, raw)

    bits = _as_tuple(tags["bits_per_sample"])
    sample_format = _as_tuple(tags.get("sample_format", 1))
    dtype = _dtype_from_tags(bits, sample_format)
    meta = TiffMeta(
        width=int(tags["width"]),
        height=int(tags["height"]),
        samples_per_pixel=int(tags.get("samples_per_pixel", 1)),
        bits_per_sample=bits,
        sample_format=sample_format,
        compression=int(tags.get("compression", 1)),
        rows_per_strip=int(tags.get("rows_per_strip", tags["height"])),
        strip_offsets=_as_tuple(tags["strip_offsets"]),
        strip_byte_counts=_as_tuple(tags["strip_byte_counts"]),
        dtype=str(dtype),
        endian=endian,
    )
    return data, meta


def read_tiff_array(path: Path) -> tuple[np.ndarray, TiffMeta]:
    data, meta = read_tiff_tags(path)
    dtype = np.dtype(meta.dtype)

    chunks: list[bytes] = []
    for offset, byte_count in zip(meta.strip_offsets, meta.strip_byte_counts):
        chunk = data[offset : offset + byte_count]
        if meta.compression == 1:
            chunks.append(chunk)
        elif meta.compression == 8:
            chunks.append(zlib.decompress(chunk))
        else:
            raise ValueError(f"Unsupported TIFF compression={meta.compression}: {path}")

    raw = b"".join(chunks)
    arr = np.frombuffer(raw, dtype=dtype.newbyteorder(meta.endian))
    expected = meta.width * meta.height * meta.samples_per_pixel
    if arr.size != expected:
        raise ValueError(f"Unexpected TIFF size {arr.size} != {expected}: {path}")

    if meta.samples_per_pixel == 1:
        arr = arr.reshape(meta.height, meta.width)
    else:
        arr = arr.reshape(meta.height, meta.width, meta.samples_per_pixel)
    return arr.astype(dtype, copy=False), meta


def write_float32_like_template(template_path: Path, output_path: Path, array: np.ndarray) -> None:
    """Write a single-channel float32 TIFF by replacing template raster bytes."""

    data, meta = read_tiff_tags(template_path)
    if meta.compression != 1:
        raise ValueError(f"Template must be uncompressed: {template_path}")
    if meta.samples_per_pixel != 1 or meta.dtype != "float32":
        raise ValueError(f"Template must be single-channel float32: {template_path}")

    arr = np.asarray(array, dtype=np.float32)
    if arr.shape != (meta.height, meta.width):
        raise ValueError(f"Prediction shape {arr.shape} != template shape {(meta.height, meta.width)}")

    raw = arr.astype(np.dtype("float32").newbyteorder(meta.endian), copy=False).tobytes(order="C")
    expected_bytes = sum(meta.strip_byte_counts)
    if len(raw) != expected_bytes:
        raise ValueError(f"Prediction bytes {len(raw)} != template bytes {expected_bytes}")

    out = bytearray(data)
    cursor = 0
    for offset, byte_count in zip(meta.strip_offsets, meta.strip_byte_counts):
        out[offset : offset + byte_count] = raw[cursor : cursor + byte_count]
        cursor += byte_count

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(out)

