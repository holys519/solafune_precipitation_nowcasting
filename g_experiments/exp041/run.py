#!/usr/bin/env python3
"""exp041: 5-source OOF-weighted blend (exp016/017/018/exp035_no_dilation + exp038_features)
+ joint post-processing (5-tap +-60min smoothing, blur, per-satellite thresholds) + overlap
patch.

exp038_features is architecturally distinct from the other four (current-row-only, no
successor rows; wavelength-aligned physics features instead) -- a genuine diversity source
rather than a seed/hyperparameter variant of the same family.

Weights: g_eda/exp003's 4way_simplex_result.json (fixed 4-way per-satellite optimum)
combined with 5source_recommendation.json's per-satellite exp038_features blend-in weight:
    w_model_final = (1 - w_new) * w_model_4way   for model in the 4-way family
    w_exp038_features_final = w_new
Post-processing: g_eda/exp004's recommended_postprocess.json (5-tap smoothing, blur,
per-satellite value thresholds), same stacking order as exp036/exp039 (blend -> smooth ->
blur -> threshold -> patch last).
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP014 = ROOT / "g_experiments/exp014"
EXP017 = ROOT / "g_experiments/exp017"
SUBMISSIONS = ROOT / "outputs/submissions"
ANALYSIS_DIR = ROOT / "outputs/analysis/exp041"
EVALUATION_CSV = ROOT / "data/evaluation_dataset/evaluation_target.csv"
TRAIN_CSV = ROOT / "data/train_dataset/train_dataset.csv"
TRAIN_DIR = ROOT / "data/train_dataset"

FOUR_WAY_SIMPLEX = ROOT / "outputs/g_eda/exp003/4way_simplex_result.json"
FIVE_SOURCE = ROOT / "outputs/g_eda/exp003/5source_recommendation.json"
POSTPROCESS = ROOT / "outputs/g_eda/exp004/recommended_postprocess.json"

FOUR_WAY = ("exp016", "exp017", "exp018", "exp035_no_dilation")
MODELS = (*FOUR_WAY, "exp038_features")
SATELLITES = ("goes", "himawari", "meteosat")

sys.path.insert(0, str(EXP017))
from tiff_utils import read_tiff_array, write_float32_like_template  # noqa: E402


def read_evaluation_rows() -> list[dict[str, str]]:
    with EVALUATION_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    names = [row["gpm_imerg_filename"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("duplicate gpm_imerg_filename values")
    return rows


def source_files(source_dir: Path) -> dict[str, Path]:
    test_files = source_dir / "test_files"
    if not test_files.is_dir():
        raise FileNotFoundError(f"missing source prediction directory: {test_files}")
    return {path.name: path for path in test_files.glob("*.tif")}


def final_weights() -> dict[str, dict[str, float]]:
    four_way = json.loads(FOUR_WAY_SIMPLEX.read_text())["per_satellite_best"]
    w_new = json.loads(FIVE_SOURCE.read_text())["per_satellite_new_source_weight"]
    out = {}
    for sat in SATELLITES:
        w16, w17, w18, wnd = four_way[sat]["weights"]
        new = w_new[sat]
        scale = 1.0 - new
        out[sat] = {
            "exp016": scale * w16,
            "exp017": scale * w17,
            "exp018": scale * w18,
            "exp035_no_dilation": scale * wnd,
            "exp038_features": new,
        }
        total = sum(out[sat].values())
        assert 0.999 <= total <= 1.001, (sat, total)
    return out


def gaussian_blur_2d(array: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(array, ((radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(array)
    for i, k in enumerate(kernel):
        out += k * padded[i : i + array.shape[0], :]
    padded = np.pad(out, ((0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(array)
    for i, k in enumerate(kernel):
        out += k * padded[:, i : i + array.shape[1]]
    return out


def neighbor_indices(rows: list[dict[str, str]]) -> dict[int, list[int]]:
    key_of = {}
    for i, row in enumerate(rows):
        key_of[(row["name_location"], datetime.fromisoformat(row["datetime"]))] = i
    neighbors: dict[int, list[int]] = {}
    for offset in (-60, -30, 30, 60):
        idx = []
        for row in rows:
            when = datetime.fromisoformat(row["datetime"])
            idx.append(key_of.get((row["name_location"], when + timedelta(minutes=offset)), -1))
        neighbors[offset] = idx
    return neighbors


def build(name: str, weights: dict[str, dict[str, float]], rows: list[dict[str, str]],
          files: dict[str, dict[str, Path]], postprocess: dict) -> Path:
    raw_dir = SUBMISSIONS / f"exp041/{name}_raw"
    destination = raw_dir / "test_files"
    destination.mkdir(parents=True, exist_ok=True)

    blended_arrays: list[np.ndarray] = []
    templates: list[Path] = []
    for index, row in enumerate(rows, start=1):
        filename = row["gpm_imerg_filename"]
        triple = weights[row["satellite_target"]]
        blended = None
        template = None
        for model in MODELS:
            weight = triple[model]
            if weight == 0.0:
                continue
            array, _ = read_tiff_array(files[model][filename])
            template = template or files[model][filename]
            contribution = weight * array.astype(np.float32)
            blended = contribution if blended is None else blended + contribution
        blended_arrays.append(np.maximum(blended, 0.0))
        templates.append(template)
        if index % 5000 == 0 or index == len(rows):
            print(f"{name}: blended {index}/{len(rows)}", flush=True)

    neighbors = neighbor_indices(rows)
    smooth_weights = postprocess["per_satellite_smooth"]
    smoothed = []
    for i, row in enumerate(rows):
        cw, p1, n1, p2, n2 = smooth_weights[row["satellite_target"]]
        weighted = cw * blended_arrays[i]
        total = cw
        for offset, w in ((-30, p1), (30, n1), (-60, p2), (60, n2)):
            if w <= 0.0:
                continue
            j = neighbors[offset][i]
            if j >= 0:
                weighted = weighted + w * blended_arrays[j]
                total += w
        smoothed.append(weighted / total)

    blur_sigma = float(postprocess["blur_sigma"])
    thresholds = postprocess["per_satellite_thresholds"]
    for i, row in enumerate(rows):
        array = smoothed[i]
        if blur_sigma > 0.0:
            array = gaussian_blur_2d(array, blur_sigma)
        threshold = float(thresholds[row["satellite_target"]])
        if threshold > 0.0:
            array = np.where(array < threshold, 0.0, array)
        write_float32_like_template(templates[i], destination / row["gpm_imerg_filename"], array)
    shutil.copy2(EVALUATION_CSV, raw_dir / "evaluation_target.csv")
    return raw_dir


def create_submission_zip(source_dir: Path, zip_path: Path, filenames: list[str]) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        archive.write(source_dir / "evaluation_target.csv", "evaluation_target.csv")
        for filename in filenames:
            archive.write(source_dir / "test_files" / filename, f"test_files/{filename}")
    validate_submission_zip(zip_path, filenames)
    return zip_path


def validate_submission_zip(zip_path: Path, filenames: list[str]) -> None:
    expected = {"evaluation_target.csv", *(f"test_files/{name}" for name in filenames)}
    with zipfile.ZipFile(zip_path) as archive:
        names = archive.namelist()
        bad = archive.testzip()
    if len(names) != len(set(names)) or set(names) != expected:
        raise ValueError(f"zip file-set mismatch for {zip_path}")
    if bad is not None:
        raise ValueError(f"corrupt entry in {zip_path}: {bad}")


def apply_overlap_patch(name: str, raw_dir: Path, filenames: list[str]) -> Path:
    patched_dir = SUBMISSIONS / f"exp041/{name}_patched"
    patched_zip = SUBMISSIONS / f"exp041_{name}_patched.zip"
    patch_config = {
        "experiment": {"name": "exp041",
                       "description": f"4-source OOF blend {name} + exp014 overlap patch",
                       "seed": 42},
        "data": {"train_csv": str(TRAIN_CSV), "evaluation_csv": str(EVALUATION_CSV),
                 "train_dir": str(TRAIN_DIR),
                 "evaluation_dir": str(ROOT / "data/evaluation_dataset")},
        "overlap": {"pairs_table": str(EXP014 / "overlap_pairs.csv"), "min_agreement": 0.0},
        "paths": {"source_submission_dir": str(raw_dir), "output_dir": str(patched_dir),
                  "submission_zip": str(patched_zip)},
    }
    config_path = ANALYSIS_DIR / f"patch_{name}.yaml"
    config_path.write_text(yaml.safe_dump(patch_config, sort_keys=False), encoding="utf-8")
    subprocess.run([sys.executable, str(EXP014 / "apply_overlap.py"), "--config", str(config_path)],
                   check=True)
    validate_submission_zip(patched_zip, filenames)
    return patched_zip


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_evaluation_rows()
    filenames = [row["gpm_imerg_filename"] for row in rows]
    files = {model: source_files(SUBMISSIONS / model) for model in MODELS}
    for model, model_files in files.items():
        missing = set(filenames) - set(model_files)
        if missing:
            raise ValueError(f"{model}: {len(missing)} evaluation files missing")

    weights = final_weights()
    postprocess = json.loads(POSTPROCESS.read_text())
    print(json.dumps({"weights": weights, "postprocess": postprocess}, indent=2), flush=True)

    name = "5src_joint"
    raw_dir = build(name, weights, rows, files, postprocess)
    raw_zip = create_submission_zip(raw_dir, SUBMISSIONS / f"exp041_{name}_raw.zip", filenames)
    patched_zip = apply_overlap_patch(name, raw_dir, filenames)

    summary = {
        "experiment": "exp041", "weights": weights, "postprocess": postprocess,
        "raw_zip": str(raw_zip), "raw_zip_sha256": sha256(raw_zip),
        "patched_zip": str(patched_zip), "patched_zip_sha256": sha256(patched_zip),
    }
    (ANALYSIS_DIR / f"analysis_summary_{name}.json").write_text(json.dumps(summary, indent=2),
                                                                encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
