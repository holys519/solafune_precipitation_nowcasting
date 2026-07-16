#!/usr/bin/env python3
"""exp036: OOF-optimized exp016/017/018 blend (optionally per-satellite) + overlap patch.

Successor to exp033's fixed-ratio ladder. Weights come from g_eda/exp003's
recommended_weights.json (OOF simplex search); pass --scheme to pick which recommendation
to serve, or --weights to override manually.

Schemes:
- global:        one (w016, w017, w018) triple for every tile (global_best)
- per_satellite: each satellite uses its own OOF-optimal triple (per_satellite_best)

Blend happens on the raw eval predictions of exp016/exp017/exp018, then the exp014 overlap
patch is applied last (same order as exp026/exp033 — patching first would let the blend
dilute known ground-truth pixels).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import yaml
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parents[2]
EXP014 = ROOT / "g_experiments/exp014"
EXP017 = ROOT / "g_experiments/exp017"
SUBMISSIONS = ROOT / "outputs/submissions"
# Overridden by --out-prefix / --sources-root (exp037 reuses this pipeline for TTA sources).
EXP_PREFIX = "exp036"
ANALYSIS_DIR = ROOT / "outputs/analysis/exp036"
EVALUATION_CSV = ROOT / "data/evaluation_dataset/evaluation_target.csv"
TRAIN_CSV = ROOT / "data/train_dataset/train_dataset.csv"
TRAIN_DIR = ROOT / "data/train_dataset"
RECOMMENDED = ROOT / "outputs/g_eda/exp003/recommended_weights.json"

MODELS = ("exp016", "exp017", "exp018")
SATELLITES = ("goes", "himawari", "meteosat")

sys.path.insert(0, str(EXP017))
from tiff_utils import read_tiff_array, write_float32_like_template  # noqa: E402


def read_evaluation_rows() -> list[dict[str, str]]:
    with EVALUATION_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    names = [row["gpm_imerg_filename"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("evaluation CSV contains duplicate gpm_imerg_filename values")
    return rows


def source_files(source_dir: Path) -> dict[str, Path]:
    test_files = source_dir / "test_files"
    if not test_files.is_dir():
        raise FileNotFoundError(f"missing source prediction directory: {test_files}")
    return {path.name: path for path in test_files.glob("*.tif")}


def load_weights(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    """Returns {satellite: {model: weight}}; the 'global' scheme repeats one triple."""
    if args.weights:
        w016, w017, w018 = (float(v) for v in args.weights.split(","))
        triple = {"exp016": w016, "exp017": w017, "exp018": w018}
        per_sat = {sat: dict(triple) for sat in SATELLITES}
    else:
        if not RECOMMENDED.exists():
            raise FileNotFoundError(f"{RECOMMENDED} not found — run g_eda/exp003 first or pass --weights")
        rec = json.loads(RECOMMENDED.read_text())
        if args.scheme == "global":
            best = rec["global_best"]
            triple = {"exp016": best["w016"], "exp017": best["w017"], "exp018": best["w018"]}
            per_sat = {sat: dict(triple) for sat in SATELLITES}
        else:
            per_sat = {
                sat: {"exp016": rec["per_satellite_best"][sat]["w016"],
                      "exp017": rec["per_satellite_best"][sat]["w017"],
                      "exp018": rec["per_satellite_best"][sat]["w018"]}
                for sat in SATELLITES
            }
    for sat, triple in per_sat.items():
        total = sum(triple.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"{sat}: weights must sum to 1, got {total}")
    return per_sat


def gaussian_blur_2d(array: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian with edge padding — matches g_eda/exp003's OOF sweep exactly."""
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
    """Same-location temporal neighbors at +-30/+-60 min, mirroring the g_eda/exp004 sweeps."""
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


def blend(name: str, weights: dict[str, dict[str, float]], rows: list[dict[str, str]],
          files: dict[str, dict[str, Path]], blur_sigma: float = 0.0,
          value_threshold: float = 0.0,
          smooth: tuple[float, float, float] | None = None,
          per_sat_smooth: dict[str, list[float]] | None = None,
          per_sat_thresholds: dict[str, float] | None = None) -> Path:
    raw_dir = SUBMISSIONS / f"{EXP_PREFIX}/{name}_raw"
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

    # Stacking order matches the OOF sweeps: blend -> temporal smoothing -> blur -> threshold
    # (patch last).
    if smooth is not None or per_sat_smooth is not None:
        neighbors = neighbor_indices(rows)
        smoothed = []
        for i, row in enumerate(rows):
            if per_sat_smooth is not None:
                cw, p1, n1, p2, n2 = per_sat_smooth[row["satellite_target"]]
            else:
                cw, p1, n1 = smooth
                p2 = n2 = 0.0
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
        blended_arrays = smoothed

    for i, row in enumerate(rows):
        array = blended_arrays[i]
        if blur_sigma > 0.0:
            array = gaussian_blur_2d(array, blur_sigma)
        threshold = value_threshold
        if per_sat_thresholds is not None:
            threshold = float(per_sat_thresholds[row["satellite_target"]])
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
    patched_dir = SUBMISSIONS / f"{EXP_PREFIX}/{name}_patched"
    patched_zip = SUBMISSIONS / f"{EXP_PREFIX}_{name}_patched.zip"
    patch_config = {
        "experiment": {"name": EXP_PREFIX,
                       "description": f"OOF-weighted blend {name} + exp014 overlap patch",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["global", "per_satellite"], default="global")
    parser.add_argument("--weights", help="manual override: 'w016,w017,w018' (global scheme)")
    parser.add_argument("--blur-sigma", type=float, default=0.0,
                        help="gaussian blur applied after blending (OOF combo sweep value)")
    parser.add_argument("--value-threshold", type=float, default=0.0,
                        help="zero out pixels below this after blur (OOF combo sweep value)")
    parser.add_argument("--smooth", default=None,
                        help="temporal smoothing 'center,prev,next' (g_eda/exp004 sweep value)")
    parser.add_argument("--postprocess-json", default=None,
                        help="g_eda/exp004 recommended_postprocess.json: per-satellite 5-tap "
                             "smoothing + blur + per-satellite thresholds (overrides "
                             "--smooth/--blur-sigma/--value-threshold)")
    parser.add_argument("--zip-raw", action="store_true", help="also zip the unpatched blend")
    parser.add_argument("--skip-patch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sources-root", default=str(SUBMISSIONS),
                        help="root holding exp016/exp017/exp018 prediction dirs")
    parser.add_argument("--out-prefix", default="exp036",
                        help="prefix for output dirs/zips (e.g. exp037 for TTA sources)")
    args = parser.parse_args()

    global EXP_PREFIX, ANALYSIS_DIR
    EXP_PREFIX = args.out_prefix
    ANALYSIS_DIR = ROOT / "outputs/analysis" / EXP_PREFIX
    sources_root = Path(args.sources_root)

    rows = read_evaluation_rows()
    filenames = [row["gpm_imerg_filename"] for row in rows]
    files = {model: source_files(sources_root / model) for model in MODELS}
    for model, model_files in files.items():
        missing = set(filenames) - set(model_files)
        if missing:
            raise ValueError(f"{model}: {len(missing)} evaluation files missing")

    weights = load_weights(args)
    name = args.scheme if not args.weights else "manual"
    per_sat_smooth = None
    per_sat_thresholds = None
    if args.postprocess_json:
        post = json.loads(Path(args.postprocess_json).read_text())
        per_sat_smooth = post["per_satellite_smooth"]
        per_sat_thresholds = post["per_satellite_thresholds"]
        args.blur_sigma = float(post["blur_sigma"])
        args.smooth = None
        name_suffix = "_joint"
    else:
        name_suffix = ""
    smooth = None
    if args.smooth:
        smooth = tuple(float(v) for v in args.smooth.split(","))
        if len(smooth) != 3:
            raise ValueError("--smooth expects 'center,prev,next'")
        name += f"_sm{smooth[0]:g}".replace(".", "p")
    if args.blur_sigma > 0.0:
        name += f"_blur{args.blur_sigma:g}".replace(".", "p")
    if args.value_threshold > 0.0 and per_sat_thresholds is None:
        name += f"_thr{args.value_threshold:g}".replace(".", "p")
    name += name_suffix
    print(json.dumps({"scheme": name, "weights": weights, "blur_sigma": args.blur_sigma,
                      "value_threshold": args.value_threshold, "files": len(filenames)},
                     indent=2), flush=True)
    if args.dry_run:
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = blend(name, weights, rows, files, blur_sigma=args.blur_sigma,
                    value_threshold=args.value_threshold, smooth=smooth,
                    per_sat_smooth=per_sat_smooth, per_sat_thresholds=per_sat_thresholds)
    entry: dict[str, object] = {"scheme": name, "weights": weights,
                                "blur_sigma": args.blur_sigma,
                                "value_threshold": args.value_threshold,
                                "smooth": list(smooth) if smooth else None,
                                "per_sat_smooth": per_sat_smooth,
                                "per_sat_thresholds": per_sat_thresholds,
                                "files": len(filenames), "raw_dir": str(raw_dir)}
    if args.zip_raw:
        raw_zip = create_submission_zip(raw_dir, SUBMISSIONS / f"{EXP_PREFIX}_{name}_raw.zip", filenames)
        entry["raw_zip"] = str(raw_zip)
        entry["raw_zip_sha256"] = sha256(raw_zip)
    if not args.skip_patch:
        patched_zip = apply_overlap_patch(name, raw_dir, filenames)
        entry["patched_zip"] = str(patched_zip)
        entry["patched_zip_sha256"] = sha256(patched_zip)
        entry["patched_zip_bytes"] = patched_zip.stat().st_size

    summary_path = ANALYSIS_DIR / f"analysis_summary_{name}.json"
    summary_path.write_text(json.dumps({"experiment": EXP_PREFIX, **entry}, indent=2),
                            encoding="utf-8")
    print(f"wrote manifest: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
