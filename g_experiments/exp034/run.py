#!/usr/bin/env python3
"""OOF-selected rain-threshold inference, blending, and optional overlap patch.

The original exp016/017/018 submissions used rain_prob_threshold=0 and
value_threshold=0.10.  Their OOF sweeps selected rain-probability thresholds
0.25, 0.70, and 0.40 respectively.  This experiment re-runs only evaluation
inference with those fixed thresholds and value_threshold=0, builds a blend
ladder, then applies the same overlap patch as exp026.
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
EXPERIMENTS = ROOT / "g_experiments"
EXP014 = EXPERIMENTS / "exp014"
EXP017 = EXPERIMENTS / "exp017"
SUBMISSIONS = ROOT / "outputs/submissions"
ANALYSIS_DIR = ROOT / "outputs/analysis/exp034"
EVALUATION_CSV = ROOT / "data/evaluation_dataset/evaluation_target.csv"
TRAIN_CSV = ROOT / "data/train_dataset/train_dataset.csv"
TRAIN_DIR = ROOT / "data/train_dataset"

MODEL_SPECS = {
    "exp016": {"rain_prob_threshold": 0.25},
    "exp017": {"rain_prob_threshold": 0.70},
    "exp018": {"rain_prob_threshold": 0.40},
}
DEFAULT_EXP018_WEIGHTS = (0.00, 0.25, 0.50, 1.00)

sys.path.insert(0, str(EXP017))
from tiff_utils import read_tiff_array, write_float32_like_template  # noqa: E402


def read_evaluation_filenames() -> list[str]:
    with EVALUATION_CSV.open(newline="") as f:
        names = [row["gpm_imerg_filename"] for row in csv.DictReader(f)]
    if len(names) != len(set(names)):
        raise ValueError("evaluation CSV contains duplicate output filenames")
    return names


def parse_weights(raw: str) -> list[float]:
    weights = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not weights or any(weight < 0.0 or weight > 1.0 for weight in weights):
        raise ValueError(f"exp018 weights must be a non-empty list in [0,1]: {weights}")
    if len(weights) != len(set(weights)):
        raise ValueError(f"duplicate weights are not allowed: {weights}")
    return weights


def candidate_name(weight: float) -> str:
    return f"thr_w018_{int(round(weight * 100)):03d}"


def prediction_dir(model_name: str) -> Path:
    return SUBMISSIONS / f"exp034/{model_name}_thresholded"


def has_complete_predictions(directory: Path, expected: set[str]) -> bool:
    test_files = directory / "test_files"
    return test_files.is_dir() and {p.name for p in test_files.glob("*.tif")} == expected


def build_inference_config(model_name: str) -> Path:
    spec = MODEL_SPECS[model_name]
    source_config = EXPERIMENTS / model_name / "config.yaml"
    config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    output_dir = prediction_dir(model_name)
    model_analysis_dir = ANALYSIS_DIR / f"{model_name}_thresholded"

    postprocess = config.setdefault("postprocess", {})
    postprocess["use_oof_calibration"] = False
    postprocess["rain_prob_threshold"] = float(spec["rain_prob_threshold"])
    postprocess["value_threshold"] = 0.0
    postprocess.setdefault("temporal_smoothing", {})["enabled"] = False

    config["paths"]["output_dir"] = str(output_dir)
    config["paths"]["analysis_dir"] = str(model_analysis_dir)
    config["paths"]["submission_zip"] = str(
        SUBMISSIONS / f"exp034_{model_name}_thresholded.zip"
    )

    model_analysis_dir.mkdir(parents=True, exist_ok=True)
    config_path = model_analysis_dir / "inference_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def run_thresholded_inference(
    model_name: str, expected_names: set[str], force: bool
) -> Path:
    output_dir = prediction_dir(model_name)
    if not force and has_complete_predictions(output_dir, expected_names):
        print(f"{model_name}: complete thresholded predictions already exist; skipping", flush=True)
        return output_dir

    checkpoints = sorted((ROOT / f"g_model/{model_name}").glob("best_model_fold*.pt"))
    if len(checkpoints) != 5:
        raise FileNotFoundError(
            f"{model_name}: expected 5 fold checkpoints, found {len(checkpoints)}"
        )
    config_path = build_inference_config(model_name)
    command = [
        sys.executable,
        str(EXPERIMENTS / model_name / "inference.py"),
        "--config",
        str(config_path),
    ]
    for checkpoint in checkpoints:
        command.extend(("--checkpoint", str(checkpoint)))

    threshold = MODEL_SPECS[model_name]["rain_prob_threshold"]
    print(f"{model_name}: inference rain_prob_threshold={threshold} value_threshold=0", flush=True)
    subprocess.run(command, check=True)
    if not has_complete_predictions(output_dir, expected_names):
        raise RuntimeError(f"{model_name}: inference did not produce the complete evaluation set")
    return output_dir


def file_map(source_dir: Path, expected_names: set[str]) -> dict[str, Path]:
    files = {path.name: path for path in (source_dir / "test_files").glob("*.tif")}
    if set(files) != expected_names:
        raise ValueError(
            f"prediction file-set mismatch under {source_dir}: "
            f"missing={len(expected_names - set(files))} extra={len(set(files) - expected_names)}"
        )
    return files


def blend_thresholded_models(
    name: str,
    exp018_weight: float,
    filenames: list[str],
    sources: dict[str, Path],
) -> Path:
    expected_names = set(filenames)
    files = {model: file_map(path, expected_names) for model, path in sources.items()}
    # The non-exp018 half remains the LB-winning equal exp016/exp017 balance.
    weights = {
        "exp016": (1.0 - exp018_weight) * 0.5,
        "exp017": (1.0 - exp018_weight) * 0.5,
        "exp018": exp018_weight,
    }
    raw_dir = SUBMISSIONS / f"exp034/{name}_raw"
    destination = raw_dir / "test_files"
    destination.mkdir(parents=True, exist_ok=True)

    for index, filename in enumerate(filenames, start=1):
        active = [(model, weight) for model, weight in weights.items() if weight > 0.0]
        reference = files[active[0][0]][filename]
        if len(active) == 1 and active[0][1] == 1.0:
            shutil.copyfile(reference, destination / filename)
        else:
            blended = sum(
                weight * read_tiff_array(files[model][filename])[0].astype(np.float32)
                for model, weight in active
            )
            write_float32_like_template(
                reference, destination / filename, np.maximum(blended, 0.0)
            )
        if index % 5000 == 0 or index == len(filenames):
            print(f"{name}: blended {index}/{len(filenames)}", flush=True)

    shutil.copy2(EVALUATION_CSV, raw_dir / "evaluation_target.csv")
    if not has_complete_predictions(raw_dir, expected_names):
        raise RuntimeError(f"{name}: incomplete blend output")
    return raw_dir


def validate_zip(zip_path: Path, filenames: list[str]) -> None:
    expected = {"evaluation_target.csv", *(f"test_files/{name}" for name in filenames)}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        bad = archive.testzip()
    actual = set(names)
    if len(names) != len(actual) or actual != expected or bad is not None:
        raise ValueError(
            f"invalid submission zip {zip_path}: duplicate={len(names) != len(actual)} "
            f"missing={len(expected - actual)} extra={len(actual - expected)} bad={bad}"
        )


def create_zip(source_dir: Path, zip_path: Path, filenames: list[str]) -> Path:
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1
    ) as archive:
        archive.write(source_dir / "evaluation_target.csv", "evaluation_target.csv")
        for filename in filenames:
            archive.write(source_dir / "test_files" / filename, f"test_files/{filename}")
    validate_zip(zip_path, filenames)
    return zip_path


def apply_overlap_patch(name: str, raw_dir: Path, filenames: list[str]) -> Path:
    output_dir = SUBMISSIONS / f"exp034/{name}_patched"
    zip_path = SUBMISSIONS / f"exp034_{name}_patched.zip"
    config = {
        "experiment": {
            "name": "exp034",
            "description": f"OOF rain-threshold blend {name} with exp014 overlap patch",
            "seed": 42,
        },
        "data": {
            "train_csv": str(TRAIN_CSV),
            "evaluation_csv": str(EVALUATION_CSV),
            "train_dir": str(TRAIN_DIR),
            "evaluation_dir": str(ROOT / "data/evaluation_dataset"),
        },
        "overlap": {"pairs_table": str(EXP014 / "overlap_pairs.csv"), "min_agreement": 0.0},
        "paths": {
            "source_submission_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "submission_zip": str(zip_path),
        },
    }
    config_path = ANALYSIS_DIR / f"patch_{name}.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    subprocess.run(
        [sys.executable, str(EXP014 / "apply_overlap.py"), "--config", str(config_path)],
        check=True,
    )
    validate_zip(zip_path, filenames)
    return zip_path


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
        default=",".join(str(weight) for weight in DEFAULT_EXP018_WEIGHTS),
        help="exp018 blend weights; default: 0.00,0.25,0.50,1.00",
    )
    parser.add_argument("--force-inference", action="store_true")
    parser.add_argument("--skip-inference", action="store_true")
    parser.add_argument("--skip-patch", action="store_true")
    parser.add_argument("--zip-raw", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    weights = parse_weights(args.weights)
    filenames = read_evaluation_filenames()
    expected_names = set(filenames)
    plan = {
        "files": len(filenames),
        "thresholds": {
            model: spec["rain_prob_threshold"] for model, spec in MODEL_SPECS.items()
        },
        "value_threshold": 0.0,
        "exp018_blend_weights": weights,
    }
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        # Dry-run verifies checkpoints and any existing prediction directories without GPU inference.
        for model_name in MODEL_SPECS:
            checkpoints = sorted((ROOT / f"g_model/{model_name}").glob("best_model_fold*.pt"))
            if len(checkpoints) != 5:
                raise FileNotFoundError(
                    f"{model_name}: expected 5 checkpoints, found {len(checkpoints)}"
                )
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for model_name in MODEL_SPECS:
        if args.skip_inference:
            source = prediction_dir(model_name)
            if not has_complete_predictions(source, expected_names):
                raise FileNotFoundError(
                    f"--skip-inference requested but predictions are incomplete: {source}"
                )
        else:
            source = run_thresholded_inference(model_name, expected_names, args.force_inference)
        sources[model_name] = source

    manifest: list[dict[str, object]] = []
    for weight in weights:
        name = candidate_name(weight)
        raw_dir = blend_thresholded_models(name, weight, filenames, sources)
        entry: dict[str, object] = {
            "name": name,
            "weights": {
                "thresholded_exp016": (1.0 - weight) * 0.5,
                "thresholded_exp017": (1.0 - weight) * 0.5,
                "thresholded_exp018": weight,
            },
            "raw_dir": str(raw_dir),
            "files": len(filenames),
        }
        if args.zip_raw:
            raw_zip = create_zip(raw_dir, SUBMISSIONS / f"exp034_{name}_raw.zip", filenames)
            entry["raw_zip"] = str(raw_zip)
            entry["raw_zip_sha256"] = sha256(raw_zip)
        if not args.skip_patch:
            patched_zip = apply_overlap_patch(name, raw_dir, filenames)
            entry["patched_zip"] = str(patched_zip)
            entry["patched_zip_sha256"] = sha256(patched_zip)
        manifest.append(entry)
        print(json.dumps(entry, indent=2), flush=True)

    summary_path = ANALYSIS_DIR / "analysis_summary.json"
    summary_path.write_text(
        json.dumps({"experiment": "exp034", "plan": plan, "schemes": manifest}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote manifest: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
