#!/usr/bin/env python3
"""Build four exp018 blend-ladder submissions and optionally apply the overlap patch.

The already-scored exp026 submission is the exp018-weight=0 anchor.  This script
adds exp018 at weights 0.25, 0.50, 0.75, and 1.00 to the raw exp024
equal-exp016/exp017 blend, then applies the same exp014 overlap patch used by
exp026.  Blending always happens before patching so known overlap pixels are not
diluted by the ensemble.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXP014 = ROOT / "g_experiments/exp014"
EXP017 = ROOT / "g_experiments/exp017"
SUBMISSIONS = ROOT / "outputs/submissions"
ANALYSIS_DIR = ROOT / "outputs/analysis/exp033"
EVALUATION_CSV = ROOT / "data/evaluation_dataset/evaluation_target.csv"
TRAIN_CSV = ROOT / "data/train_dataset/train_dataset.csv"
TRAIN_DIR = ROOT / "data/train_dataset"

BASE_SOURCE = SUBMISSIONS / "exp024/equal_016_017"
EXP018_SOURCE = SUBMISSIONS / "exp018"
DEFAULT_WEIGHTS = (0.25, 0.50, 0.75, 1.00)

sys.path.insert(0, str(EXP017))
from tiff_utils import read_tiff_array, write_float32_like_template  # noqa: E402


def parse_weights(raw: str) -> list[float]:
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not weights:
        raise ValueError("at least one exp018 weight is required")
    if len(set(weights)) != len(weights):
        raise ValueError(f"duplicate weights are not allowed: {weights}")
    if any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError(f"weights must be in [0, 1]: {weights}")
    return weights


def candidate_name(exp018_weight: float) -> str:
    return f"w018_{int(round(exp018_weight * 100)):03d}"


def read_evaluation_filenames() -> list[str]:
    with EVALUATION_CSV.open(newline="") as f:
        names = [row["gpm_imerg_filename"] for row in csv.DictReader(f)]
    if len(names) != len(set(names)):
        raise ValueError("evaluation CSV contains duplicate gpm_imerg_filename values")
    return names


def source_files(source_dir: Path) -> dict[str, Path]:
    test_files = source_dir / "test_files"
    if not test_files.is_dir():
        raise FileNotFoundError(f"missing source prediction directory: {test_files}")
    return {path.name: path for path in test_files.glob("*.tif")}


def validate_sources(expected_names: set[str]) -> tuple[dict[str, Path], dict[str, Path]]:
    base_files = source_files(BASE_SOURCE)
    exp018_files = source_files(EXP018_SOURCE)
    for label, files in (("exp024 equal_016_017", base_files), ("exp018", exp018_files)):
        missing = expected_names - set(files)
        extra = set(files) - expected_names
        if missing or extra:
            raise ValueError(
                f"{label} file-set mismatch: missing={len(missing)} extra={len(extra)}"
            )
    return base_files, exp018_files


def blend_predictions(
    name: str,
    exp018_weight: float,
    filenames: list[str],
    base_files: dict[str, Path],
    exp018_files: dict[str, Path],
) -> Path:
    raw_dir = SUBMISSIONS / f"exp033/{name}_raw"
    destination = raw_dir / "test_files"
    destination.mkdir(parents=True, exist_ok=True)

    base_weight = 1.0 - exp018_weight
    for index, filename in enumerate(filenames, start=1):
        base_path = base_files[filename]
        exp018_path = exp018_files[filename]
        output_path = destination / filename

        if exp018_weight == 1.0:
            shutil.copyfile(exp018_path, output_path)
        elif exp018_weight == 0.0:
            shutil.copyfile(base_path, output_path)
        else:
            base_array, _ = read_tiff_array(base_path)
            exp018_array, _ = read_tiff_array(exp018_path)
            blended = (
                base_weight * base_array.astype(np.float32)
                + exp018_weight * exp018_array.astype(np.float32)
            )
            write_float32_like_template(base_path, output_path, np.maximum(blended, 0.0))

        if index % 5000 == 0 or index == len(filenames):
            print(f"{name}: blended {index}/{len(filenames)}", flush=True)

    shutil.copy2(EVALUATION_CSV, raw_dir / "evaluation_target.csv")
    written = {path.name for path in destination.glob("*.tif")}
    if written != set(filenames):
        raise RuntimeError(
            f"{name}: output file-set mismatch, expected={len(filenames)} actual={len(written)}"
        )
    return raw_dir


def create_submission_zip(source_dir: Path, zip_path: Path, filenames: list[str]) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        archive.write(source_dir / "evaluation_target.csv", "evaluation_target.csv")
        for filename in filenames:
            archive.write(source_dir / "test_files" / filename, f"test_files/{filename}")
    validate_submission_zip(zip_path, filenames)
    return zip_path


def validate_submission_zip(zip_path: Path, filenames: list[str]) -> None:
    expected = {"evaluation_target.csv", *(f"test_files/{name}" for name in filenames)}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        actual = set(names)
        bad = archive.testzip()
    if len(names) != len(actual):
        raise ValueError(f"duplicate entries found in {zip_path}")
    if actual != expected:
        raise ValueError(
            f"zip file-set mismatch for {zip_path}: "
            f"missing={len(expected - actual)} extra={len(actual - expected)}"
        )
    if bad is not None:
        raise ValueError(f"corrupt entry in {zip_path}: {bad}")


def apply_overlap_patch(name: str, raw_dir: Path, filenames: list[str]) -> Path:
    patched_dir = SUBMISSIONS / f"exp033/{name}_patched"
    patched_zip = SUBMISSIONS / f"exp033_{name}_patched.zip"
    patch_config = {
        "experiment": {
            "name": "exp033",
            "description": f"exp018 blend ladder {name}, followed by exp014 overlap patch",
            "seed": 42,
        },
        "data": {
            "train_csv": str(TRAIN_CSV),
            "evaluation_csv": str(EVALUATION_CSV),
            "train_dir": str(TRAIN_DIR),
            "evaluation_dir": str(ROOT / "data/evaluation_dataset"),
        },
        "overlap": {
            "pairs_table": str(EXP014 / "overlap_pairs.csv"),
            "min_agreement": 0.0,
        },
        "paths": {
            "source_submission_dir": str(raw_dir),
            "output_dir": str(patched_dir),
            "submission_zip": str(patched_zip),
        },
    }
    config_path = ANALYSIS_DIR / f"patch_{name}.yaml"
    config_path.write_text(yaml.safe_dump(patch_config, sort_keys=False), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(EXP014 / "apply_overlap.py"), "--config", str(config_path)],
        check=True,
    )
    validate_submission_zip(patched_zip, filenames)
    return patched_zip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default=",".join(str(weight) for weight in DEFAULT_WEIGHTS),
        help="comma-separated exp018 weights; default: 0.25,0.50,0.75,1.00",
    )
    parser.add_argument(
        "--zip-raw",
        action="store_true",
        help="also create unpatched submission zips for the rule-safe fallback",
    )
    parser.add_argument(
        "--skip-patch",
        action="store_true",
        help="build only unpatched predictions (use with --zip-raw to create zips)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate source coverage and print the plan without writing predictions",
    )
    args = parser.parse_args()

    weights = parse_weights(args.weights)
    filenames = read_evaluation_filenames()
    expected_names = set(filenames)
    base_files, exp018_files = validate_sources(expected_names)

    plan = [
        {
            "name": candidate_name(weight),
            "exp024_equal_016_017_weight": 1.0 - weight,
            "exp018_weight": weight,
        }
        for weight in weights
    ]
    print(json.dumps({"files": len(filenames), "plan": plan}, indent=2), flush=True)
    if args.dry_run:
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    for item in plan:
        name = str(item["name"])
        exp018_weight = float(item["exp018_weight"])
        raw_dir = blend_predictions(
            name, exp018_weight, filenames, base_files, exp018_files
        )
        entry: dict[str, object] = {**item, "files": len(filenames), "raw_dir": str(raw_dir)}

        if args.zip_raw:
            raw_zip = create_submission_zip(
                raw_dir, SUBMISSIONS / f"exp033_{name}_raw.zip", filenames
            )
            entry["raw_zip"] = str(raw_zip)
            entry["raw_zip_sha256"] = sha256(raw_zip)

        if not args.skip_patch:
            patched_zip = apply_overlap_patch(name, raw_dir, filenames)
            entry["patched_zip"] = str(patched_zip)
            entry["patched_zip_sha256"] = sha256(patched_zip)
            entry["patched_zip_bytes"] = patched_zip.stat().st_size

        manifest.append(entry)
        print(json.dumps(entry, indent=2), flush=True)

    summary_path = ANALYSIS_DIR / "analysis_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "experiment": "exp033",
                "anchor": {
                    "exp018_weight": 0.0,
                    "submission": "exp026_submission.zip",
                    "public_rmse": 0.6746506841387548,
                },
                "schemes": manifest,
                "overlap_patch_applied": not args.skip_patch,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote manifest: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
