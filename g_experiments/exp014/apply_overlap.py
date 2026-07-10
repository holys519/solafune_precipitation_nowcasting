#!/usr/bin/env python3
"""exp014 (ticket G-022): overwrite eval prediction pixels with same-time train GPM truth
in the confirmed spatial-overlap regions between 3 eval tiles and their paired train tiles.

Background: `l_eda/exp002` found that north_sumatra/aceh, northeast_malaysia/hat_yai, and
sylhet/dhaka tiles spatially overlap at identical (location, satellite) timestamps -- verified
to be a bit-exact copy in the overlap region on a train-train pair (atlantic_coast/florida,
copy RMSE 0.0000). See `doc/tile_overlap_discovery.md` for the full derivation and the
offset table in `overlap_pairs.csv` (regenerable from `l_eda/exp002/run.sh`).

This is pure post-processing: it does not train or run a model. It takes an existing
submission's `test_files/` directory as the base prediction and patches only the pixels that
fall inside a confirmed overlap region, using the existing GeoTIFF as the write template (so
all metadata/dtype/shape is preserved automatically). Every other pixel is untouched.

Usage:
    python apply_overlap.py --config config.yaml
"""

from __future__ import annotations

import argparse
import csv
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from tiff_utils import read_tiff_array, write_float32_like_template

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (SCRIPT_DIR / p).resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_overlap_pairs(path: Path) -> list[dict[str, Any]]:
    pairs = []
    for row in read_csv_rows(path):
        pairs.append(
            {
                "eval_location": row["eval_location"],
                "train_location": row["train_location"],
                "satellite": row["satellite"],
                "dy": int(row["gpm_offset_dy"]),
                "dx": int(row["gpm_offset_dx"]),
            }
        )
    return pairs


def overlap_slices(shape: tuple[int, int], dy: int, dx: int) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Overlap slices matching the l_eda/exp002 `normalized_xcorr_peak` convention:
    eval[y, x] ~ train[y - dy, x - dx]. Returns (eval_slice, train_slice) such that
    eval_array[eval_slice] and train_array[train_slice] address the same physical pixels.
    """

    h, w = shape
    ey = slice(max(0, dy), h + min(0, dy))
    ex = slice(max(0, dx), w + min(0, dx))
    ty = slice(max(0, -dy), h - max(0, dy))
    tx = slice(max(0, -dx), w - max(0, dx))
    return (ey, ex), (ty, tx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()
    config = load_config(Path(args.config))

    source_dir = resolve_path(config["paths"]["source_submission_dir"])
    source_test_files = source_dir / "test_files"
    if not source_test_files.is_dir():
        raise FileNotFoundError(
            f"{source_test_files} not found -- run inference for the source experiment first "
            f"(config paths.source_submission_dir)"
        )

    train_dir = resolve_path(config["data"]["train_dir"])
    eval_csv = resolve_path(config["data"]["evaluation_csv"])
    output_dir = resolve_path(config["paths"]["output_dir"])
    zip_path = resolve_path(config["paths"]["submission_zip"])
    prediction_dir = output_dir / "test_files"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    overlap_pairs = read_overlap_pairs(resolve_path(config["overlap"]["pairs_table"]))
    overlap_by_eval_loc = {p["eval_location"]: p for p in overlap_pairs}
    min_agreement = float(config["overlap"].get("min_agreement", 0.0))
    pairs_table_rows = read_csv_rows(resolve_path(config["overlap"]["pairs_table"]))
    agreement_by_loc = {r["eval_location"]: float(r["agreement"]) for r in pairs_table_rows}
    overlap_by_eval_loc = {
        loc: pair for loc, pair in overlap_by_eval_loc.items() if agreement_by_loc.get(loc, 0.0) >= min_agreement
    }

    eval_rows = read_csv_rows(eval_csv)
    train_rows = read_csv_rows(resolve_path(config["data"]["train_csv"]))
    train_index: dict[tuple[str, str], dict[str, str]] = {
        (r["name_location"], r["datetime"]): r for r in train_rows
    }

    # Copy the CSV through unchanged; only pixel content is patched.
    (output_dir / "evaluation_target.csv").write_bytes(eval_csv.read_bytes())

    patched = 0
    skipped_no_train_row = 0
    skipped_no_source_file = 0
    copied_unchanged = 0
    patch_log: list[dict[str, Any]] = []

    for row in eval_rows:
        gpm_name = row["gpm_imerg_filename"]
        source_path = source_test_files / gpm_name
        dest_path = prediction_dir / gpm_name
        if not source_path.exists():
            skipped_no_source_file += 1
            continue

        pair = overlap_by_eval_loc.get(row["name_location"])
        if pair is None:
            shutil.copyfile(source_path, dest_path)
            copied_unchanged += 1
            continue

        train_row = train_index.get((pair["train_location"], row["datetime"]))
        if train_row is None:
            shutil.copyfile(source_path, dest_path)
            skipped_no_train_row += 1
            continue

        train_gpm_path = train_dir / "gpm_imerg" / train_row["gpm_imerg_filename"]
        if not train_gpm_path.exists():
            shutil.copyfile(source_path, dest_path)
            skipped_no_train_row += 1
            continue

        pred_array, _ = read_tiff_array(source_path)
        train_array, _ = read_tiff_array(train_gpm_path)
        pred_array = pred_array.astype(np.float32).copy()
        (ey, ex), (ty, tx) = overlap_slices(pred_array.shape, pair["dy"], pair["dx"])
        before = pred_array[ey, ex].copy()
        pred_array[ey, ex] = train_array[ty, tx]
        write_float32_like_template(source_path, dest_path, pred_array)
        patched += 1
        if len(patch_log) < 20:
            patch_log.append(
                {
                    "unique_id": row["unique_id"],
                    "name_location": row["name_location"],
                    "train_location": pair["train_location"],
                    "overlap_pixels": int((ey.stop - ey.start) * (ex.stop - ex.start)),
                    "mean_abs_change": float(np.abs(pred_array[ey, ex] - before).mean()),
                }
            )

    print(
        f"patched={patched} copied_unchanged={copied_unchanged} "
        f"skipped_no_train_row={skipped_no_train_row} skipped_no_source_file={skipped_no_source_file}"
    )
    for entry in patch_log:
        print(f"  sample patch: {entry}")

    tif_files = sorted(prediction_dir.glob("*.tif"))
    if not tif_files:
        raise FileNotFoundError(f"No tif files written under {prediction_dir}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        zf.write(output_dir / "evaluation_target.csv", "evaluation_target.csv")
        for path in tif_files:
            zf.write(path, f"test_files/{path.name}")
    print(f"created submission: {zip_path} files={len(tif_files) + 1}")


if __name__ == "__main__":
    main()
