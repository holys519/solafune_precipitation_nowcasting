#!/usr/bin/env python3
"""exp046: causal-only temporal smoothing on top of exp038's raw eval predictions.

Follows the organizers' 2026-07-20 ruling: smoothing a prediction at target time T with
model outputs for OTHER target times is only permitted if every target used is <= T
("causal"). exp036/037's temporal_smoothing (center/prev/next weights) mixed in the NEXT
row's prediction and is therefore red now; this is the causal-only rebuild (next_weight
forced to 0, weight redistributed onto center+prev) applied to exp038 (green, strict
current-row-only model). Source predictions and this smoothing step are both green.

No OOF re-tuning was done for the causal-only weights (would need a GPU re-inference pass
over OOF folds); CENTER_WEIGHT/PREV_WEIGHT below are carried over from exp036's original
bidirectional sweep with next_weight's share folded into prev_weight, and should be treated
as an untuned first attempt pending a proper OOF sweep once GPU capacity frees up.
"""

from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP038_DIR = ROOT / "g_experiments" / "exp038"

import sys  # noqa: E402

sys.path.insert(0, str(EXP038_DIR))
from tiff_utils import read_tiff_array, write_float32_like_template  # noqa: E402

SOURCE_DIR = ROOT / "outputs" / "submissions" / "exp038"
OUT_DIR = ROOT / "outputs" / "submissions" / "exp046_causal_smoothed"
ZIP_PATH = ROOT / "outputs" / "submissions" / "exp046_causal_smoothed_submission.zip"

CENTER_WEIGHT = 0.85
PREV_WEIGHT = 0.15
MAX_GAP_MINUTES = 30


def read_evaluation_rows() -> list[dict[str, str]]:
    with (SOURCE_DIR / "evaluation_target.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    names = [row["gpm_imerg_filename"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate gpm_imerg_filename values")
    return rows


def apply_causal_smoothing(rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
    by_location: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_location.setdefault(row["name_location"], []).append(idx)

    arrays: dict[int, np.ndarray] = {}
    for idx, row in enumerate(rows):
        array, _ = read_tiff_array(SOURCE_DIR / "test_files" / row["gpm_imerg_filename"])
        arrays[idx] = array.astype(np.float32)

    smoothed: dict[str, np.ndarray] = {}
    for indices in by_location.values():
        indices = sorted(indices, key=lambda i: rows[i]["datetime"])
        datetimes = [np.datetime64(rows[i]["datetime"].replace(" ", "T")) for i in indices]
        for pos, idx in enumerate(indices):
            weighted = arrays[idx] * CENTER_WEIGHT
            total_weight = CENTER_WEIGHT
            if pos > 0:
                gap = (datetimes[pos] - datetimes[pos - 1]) / np.timedelta64(1, "m")
                if 0 < gap <= MAX_GAP_MINUTES:
                    weighted = weighted + arrays[indices[pos - 1]] * PREV_WEIGHT
                    total_weight += PREV_WEIGHT
            # next-row contribution intentionally omitted -- non-causal, banned 2026-07-20.
            smoothed[rows[idx]["gpm_imerg_filename"]] = weighted / total_weight
    return smoothed


def create_submission_zip(rows: list[dict[str, str]], smoothed: dict[str, np.ndarray]) -> None:
    test_files_dir = OUT_DIR / "test_files"
    test_files_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        filename = row["gpm_imerg_filename"]
        template = SOURCE_DIR / "test_files" / filename
        write_float32_like_template(template, test_files_dir / filename, smoothed[filename])

    csv_path = OUT_DIR / "evaluation_target.csv"
    csv_path.write_bytes((SOURCE_DIR / "evaluation_target.csv").read_bytes())

    filenames = [row["gpm_imerg_filename"] for row in rows]
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(csv_path, "evaluation_target.csv")
        for filename in filenames:
            archive.write(test_files_dir / filename, f"test_files/{filename}")

    expected = {"evaluation_target.csv", *(f"test_files/{name}" for name in filenames)}
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
        bad = archive.testzip()
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError(f"zip file-set mismatch for {ZIP_PATH}")
    if bad is not None:
        raise ValueError(f"corrupt entry in {ZIP_PATH}: {bad}")


def main() -> None:
    if not (SOURCE_DIR / "test_files").is_dir():
        raise FileNotFoundError(f"missing exp038 eval predictions: {SOURCE_DIR / 'test_files'}")
    rows = read_evaluation_rows()
    smoothed = apply_causal_smoothing(rows)
    create_submission_zip(rows, smoothed)
    print(f"wrote {ZIP_PATH} ({len(rows)} files, center={CENTER_WEIGHT} prev={PREV_WEIGHT})")


if __name__ == "__main__":
    main()
