#!/usr/bin/env python3
"""exp043: all-zero eval-set baseline (pure diagnostic, no model).

Purpose: directly measure the eval-side zero-prediction RMSE. The community discussion
(ibrahimqasmi, doc/plan/discussion_all.md 2026-07 thread) proposed this as "the cleanest
probe" for train/eval distribution shift: train's own flat-zero baseline is a known number
(the community's independent oracle audit puts it at 0.746 on all 40,686 train tiles; our
own historical exp001 note of 0.962228 was on a smaller 3000-row holdout, not the full set,
so it is not directly comparable). If eval-zero comes back close to train-zero, our OOF->LB
gap is mostly genuine model-transfer loss; if eval-zero is markedly worse, a real share of
the gap is the eval sample itself being a harder (wetter) climatology mix -- independently
quantifying the E-4/E-3 regime-shift hypothesis with a model-free number.

Reuses any already-built prediction's GeoTIFF as a byte-metadata template (shape/dtype/
compression only; pixel data is overwritten with zeros here) via exp017's tiff_utils.
"""

from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP017 = ROOT / "g_experiments/exp017"
SUBMISSIONS = ROOT / "outputs/submissions"
EVALUATION_CSV = ROOT / "data/evaluation_dataset/evaluation_target.csv"
TEMPLATE_SOURCE = SUBMISSIONS / "exp042/5src_joint_raw/test_files"

sys.path.insert(0, str(EXP017))
from tiff_utils import read_tiff_array, write_float32_like_template  # noqa: E402


def main() -> None:
    with EVALUATION_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    filenames = [row["gpm_imerg_filename"] for row in rows]
    if len(filenames) != len(set(filenames)):
        raise ValueError("duplicate gpm_imerg_filename values")

    raw_dir = SUBMISSIONS / "exp043_zero_raw"
    destination = raw_dir / "test_files"
    destination.mkdir(parents=True, exist_ok=True)

    for index, filename in enumerate(filenames, start=1):
        template_path = TEMPLATE_SOURCE / filename
        array, _ = read_tiff_array(template_path)
        zeros = np.zeros_like(array, dtype=np.float32)
        write_float32_like_template(template_path, destination / filename, zeros)
        if index % 5000 == 0 or index == len(filenames):
            print(f"wrote {index}/{len(filenames)}", flush=True)

    import shutil
    shutil.copy2(EVALUATION_CSV, raw_dir / "evaluation_target.csv")

    zip_path = SUBMISSIONS / "exp043_zero_baseline.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(raw_dir / "evaluation_target.csv", "evaluation_target.csv")
        for filename in filenames:
            archive.write(destination / filename, f"test_files/{filename}")

    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        expected = {"evaluation_target.csv", *(f"test_files/{n}" for n in filenames)}
        if set(names) != expected or len(names) != len(set(names)):
            raise ValueError("zip file-set mismatch")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"corrupt entry: {bad}")

    print(f"wrote {zip_path} ({len(filenames)} files, all-zero)", flush=True)


if __name__ == "__main__":
    main()
