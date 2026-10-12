from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def _read_header_values(header_path: Path) -> dict[str, str]:
    text = header_path.read_text(encoding="ascii")
    if not text.lstrip().startswith("ENVI"):
        raise ValueError(f"Not an ENVI header: {header_path}")

    values: dict[str, str] = {}
    current_key: str | None = None
    current_value = ""

    for line in text.splitlines()[1:]:
        if current_key is not None:
            current_value += " " + line.strip()
            if "}" in line:
                values[current_key] = current_value.strip()
                current_key = None
                current_value = ""
            continue

        match = re.match(r"^\s*([^=]+?)\s*=\s*(.*?)\s*$", line)
        if not match:
            continue

        key = match.group(1).strip().lower()
        value = match.group(2).strip()
        if "{" in value and "}" not in value:
            current_key = key
            current_value = value
        else:
            values[key] = value

    if current_key is not None:
        raise ValueError(f"Unterminated value for {current_key!r} in {header_path}")
    return values


def load_envi_cube(
    header_path: str | Path,
    raw_path: str | Path,
) -> tuple[np.memmap, np.ndarray]:
    """Memory-map this project's float32, BIP-interleaved ENVI data cube."""
    header_path = Path(header_path)
    raw_path = Path(raw_path)
    if not header_path.is_file():
        raise FileNotFoundError(f"ENVI header not found: {header_path}")
    if not raw_path.is_file():
        raise FileNotFoundError(f"ENVI binary data not found: {raw_path}")

    header = _read_header_values(header_path)
    required = ("samples", "lines", "bands", "data type", "interleave", "byte order")
    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError(f"ENVI header is missing fields: {', '.join(missing)}")

    samples = int(header["samples"])
    lines = int(header["lines"])
    bands = int(header["bands"])
    if min(samples, lines, bands) <= 0:
        raise ValueError("ENVI dimensions must all be positive.")

    if int(header["data type"]) != 4:
        raise ValueError("Expected ENVI data type 4 (32-bit floating point).")
    if header["interleave"].strip().lower() != "bip":
        raise ValueError("Expected BIP interleave for this project's saved cube.")

    byte_order = int(header["byte order"])
    if byte_order not in (0, 1):
        raise ValueError(f"Unsupported ENVI byte order: {byte_order}")
    dtype = np.dtype("<f4" if byte_order == 0 else ">f4")

    expected_bytes = lines * samples * bands * dtype.itemsize
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Binary cube size mismatch: expected {expected_bytes} bytes from "
            f"the header, found {actual_bytes} bytes in {raw_path}."
        )

    wavelength_value = header.get("wavelength")
    if wavelength_value is None:
        raise ValueError("ENVI header does not contain wavelength metadata.")
    wavelength_text = wavelength_value.strip().strip("{}")
    wavelengths = np.asarray(
        [float(value) for value in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", wavelength_text)],
        dtype=float,
    )
    if wavelengths.size != bands:
        raise ValueError(
            f"Header contains {wavelengths.size} wavelengths but declares {bands} bands."
        )

    cube = np.memmap(
        raw_path,
        dtype=dtype,
        mode="r",
        shape=(lines, samples, bands),
    )
    return cube, wavelengths
