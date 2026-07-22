#!/usr/bin/env python3
"""g_experiments/exp055: green-only manifest blend submission builder.

Consumes g_eda/exp011/recommended_weights.json exactly the way g_experiments/exp036 consumed
g_eda/exp003's recommendation, but generalized to N manifest sources by name instead of a
hardcoded exp016/017/018 triple, and with two rule-compliance differences that are load-bearing,
not stylistic:

1. Every source is re-asserted green (registry_guard.assert_green) at build time, independently
   of whatever g_eda/exp011 already checked -- a hard failure here, not a comment, if a red/amber
   source ever ends up in the manifest.
2. There is no overlap-patch step anywhere in this file (unlike exp036, which applies exp014's
   patch by default). overlap patch was ruled reverse engineering and is permanently
   disqualifying (2026-07-20) -- exp055 does not import apply_overlap.py at all, so it is
   structurally impossible for a future edit to silently re-enable it here.

Post-processing is blend -> causal-only temporal smoothing (only if g_eda/exp010 has published
recommended_causal_weights.json; otherwise skipped) -> done. Never non-causal (next_weight must
be exactly 0, enforced twice: once in g_eda/exp010/causal_smoothing.py itself, once again here).

Usage:
    python3 build_submission.py --dry-run                     # show weights/sources, no I/O
    python3 build_submission.py                                # global weights, zip
    python3 build_submission.py --scheme per_satellite          # per-satellite weights
    python3 build_submission.py --skip-causal-smoothing         # blend only, no post-process
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXP011 = ROOT / "g_eda" / "exp011"
EXP010 = ROOT / "g_eda" / "exp010"
EXP038 = ROOT / "g_experiments" / "exp038"
SUBMISSIONS = ROOT / "outputs" / "submissions"
ANALYSIS_DIR = ROOT / "outputs" / "analysis" / "exp055"
EVALUATION_CSV = ROOT / "data" / "evaluation_dataset" / "evaluation_target.csv"
RECOMMENDED = ROOT / "outputs" / "g_eda" / "exp011" / "recommended_weights.json"
MANIFEST = EXP011 / "sources.json"
CAUSAL_JSON = EXP010 / "recommended_causal_weights.json"
SATELLITES = ("goes", "himawari", "meteosat")

sys.path.insert(0, str(EXP011))
import registry_guard  # noqa: E402

sys.path.insert(0, str(EXP038))
from tiff_utils import read_tiff_array, write_float32_like_template  # noqa: E402


def load_manifest() -> dict[str, dict]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {entry["name"]: entry for entry in manifest["sources"]}


def read_evaluation_rows() -> list[dict[str, str]]:
    with EVALUATION_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))
    names = [row["gpm_imerg_filename"] for row in rows]
    if len(names) != len(set(names)):
        raise ValueError("evaluation CSV contains duplicate gpm_imerg_filename values")
    return rows


def source_files(source_dir: Path) -> dict[str, Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"missing eval prediction directory: {source_dir}")
    return {path.name: path for path in source_dir.glob("*.tif")}


def load_weights(args: argparse.Namespace, manifest: dict[str, dict]) -> tuple[dict[str, dict[str, float]], str]:
    """Returns ({satellite: {source_name: weight}}, scheme_label)."""
    if not RECOMMENDED.exists():
        raise FileNotFoundError(f"{RECOMMENDED} not found -- run g_eda/exp011 first "
                                "(--cache each source, then --analyze)")
    rec = json.loads(RECOMMENDED.read_text(encoding="utf-8"))

    manifest_names = set(manifest.keys())
    rec_names = set(rec["manifest_sources"])
    if rec_names != manifest_names:
        raise ValueError(
            f"g_eda/exp011's recommendation was computed for {sorted(rec_names)} but "
            f"sources.json currently lists {sorted(manifest_names)} -- re-run "
            "`optimize_blend.py --analyze` after editing the manifest before building a submission."
        )

    if args.scheme == "global":
        triple = rec["global_best"]["weights"]
        per_sat = {sat: dict(triple) for sat in SATELLITES}
        label = "global"
    else:
        per_sat = {sat: dict(info["weights"]) for sat, info in rec["per_satellite_best"].items()}
        label = "per_satellite"

    for sat, weights in per_sat.items():
        total = sum(weights.values())
        if not 0.999 <= total <= 1.001:
            raise ValueError(f"{sat}: weights must sum to 1, got {total}: {weights}")
    return per_sat, label


def assert_all_sources_green(manifest: dict[str, dict]) -> None:
    """Hard, code-enforced refusal -- re-checked here independently of g_eda/exp011's own guard,
    against doc/submission_registry.md via registry_guard (not by convention/comment)."""
    registry_guard.assert_all_green(list(manifest.keys()))


def blend_evaluation(name: str, weights: dict[str, dict[str, float]], rows: list[dict[str, str]],
                      files: dict[str, dict[str, Path]]) -> tuple[Path, list[tuple[dict, np.ndarray, Path]]]:
    raw_dir = SUBMISSIONS / "exp055" / f"{name}_raw"
    destination = raw_dir / "test_files"
    destination.mkdir(parents=True, exist_ok=True)

    blended_items: list[tuple[dict, np.ndarray, Path]] = []
    for index, row in enumerate(rows, start=1):
        filename = row["gpm_imerg_filename"]
        triple = weights[row["satellite_target"]]
        blended = None
        template = None
        for source_name, weight in triple.items():
            if weight == 0.0:
                continue
            array, _ = read_tiff_array(files[source_name][filename])
            template = template or files[source_name][filename]
            contribution = weight * array.astype(np.float32)
            blended = contribution if blended is None else blended + contribution
        blended = np.maximum(blended, 0.0)
        blended_items.append((row, blended, template))
        if index % 5000 == 0 or index == len(rows):
            print(f"{name}: blended {index}/{len(rows)}", flush=True)
    return raw_dir, blended_items


def apply_causal_smoothing(items: list[tuple[dict, np.ndarray, Path]]) -> tuple[list[tuple[dict, np.ndarray, Path]], dict]:
    """Loads g_eda/exp010's recommended_causal_weights.json if it exists and applies its
    causal-only smoothing. Hard assertion: next_weight must be 0 -- never non-causal, checked
    here in addition to g_eda/exp010/causal_smoothing.py's own internal guard.
    """
    if not CAUSAL_JSON.exists():
        return items, {"applied": False, "reason": "g_eda/exp010/recommended_causal_weights.json "
                       "does not exist yet -- causal-only smoothing hook is wired but inactive "
                       "until that experiment publishes a recommendation."}

    sys.path.insert(0, str(EXP010))
    from causal_smoothing import apply_temporal_smoothing  # noqa: E402

    rec = json.loads(CAUSAL_JSON.read_text(encoding="utf-8"))
    smoothing_cfg = rec.get("temporal_smoothing", {})
    if float(smoothing_cfg.get("next_weight", 0.0)) != 0.0:
        raise ValueError(
            "g_eda/exp010's recommendation has next_weight != 0 -- refusing to apply a "
            "non-causal smoothing recommendation (2026-07-20 ruling forbids mixing a later "
            "target timestamp's prediction into T's prediction)."
        )
    if not smoothing_cfg.get("causal_only", False):
        raise ValueError("g_eda/exp010's recommendation does not set causal_only=true -- refusing to apply it.")

    smoothing_items = [{"name_location": row["name_location"], "datetime": row["datetime"], "array": array}
                       for row, array, _ in items]
    smoothed = apply_temporal_smoothing(smoothing_items, {"temporal_smoothing": smoothing_cfg})
    new_items = [(row, smoothed_item["array"], template)
                for (row, _, template), smoothed_item in zip(items, smoothed)]

    # blur_sigma / per_satellite_value_threshold are purely spatial/value-based (no time-direction
    # component), so they carry no causal-vs-non-causal risk -- applying them is safe regardless
    # of the temporal-smoothing ruling. They come from the same OOF-tuned recommendation.
    blur_sigma = float(rec.get("blur_sigma", 0.0) or 0.0)
    thresholds = rec.get("per_satellite_value_threshold") or {}
    if blur_sigma > 0.0 or thresholds:
        final_items = []
        for row, array, template in new_items:
            if blur_sigma > 0.0:
                array = gaussian_blur_2d(array, blur_sigma)
            threshold = float(thresholds.get(row["satellite_target"], 0.0))
            if threshold > 0.0:
                array = np.where(array < threshold, 0.0, array)
            final_items.append((row, array, template))
        new_items = final_items

    return new_items, {"applied": True, "source_experiment": rec.get("source_experiment"),
                       "temporal_smoothing": smoothing_cfg, "blur_sigma": blur_sigma,
                       "per_satellite_value_threshold": thresholds}


def gaussian_blur_2d(array: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian with edge padding -- matches g_eda/exp003/exp010's sweep exactly."""
    import math
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


def write_predictions(raw_dir: Path, items: list[tuple[dict, np.ndarray, Path]]) -> None:
    destination = raw_dir / "test_files"
    destination.mkdir(parents=True, exist_ok=True)
    for row, array, template in items:
        write_float32_like_template(template, destination / row["gpm_imerg_filename"], array)
    import shutil
    shutil.copy2(EVALUATION_CSV, raw_dir / "evaluation_target.csv")


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scheme", choices=["global", "per_satellite"], default="global")
    parser.add_argument("--skip-causal-smoothing", action="store_true",
                        help="build the raw blend only, skip g_eda/exp010's causal smoothing hook")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest()
    # Hard, code-enforced refusal -- not just a comment. Raises RegistryComplianceError and
    # aborts before any red/amber source's predictions are ever opened.
    assert_all_sources_green(manifest)

    rows = read_evaluation_rows()
    filenames = [row["gpm_imerg_filename"] for row in rows]
    files = {}
    for name, entry in manifest.items():
        pred_dir = ROOT / entry["eval_pred_dir"]
        files[name] = source_files(pred_dir)
        missing = set(filenames) - set(files[name])
        if missing:
            raise ValueError(f"{name}: {len(missing)} evaluation files missing under {pred_dir}")

    weights, scheme_label = load_weights(args, manifest)
    name = f"{scheme_label}_blend"

    print(json.dumps({"scheme": name, "weights": weights, "sources": list(manifest.keys()),
                      "files": len(filenames)}, indent=2), flush=True)
    if args.dry_run:
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir, items = blend_evaluation(name, weights, rows, files)

    causal_info = {"applied": False, "reason": "--skip-causal-smoothing passed"}
    if not args.skip_causal_smoothing:
        items, causal_info = apply_causal_smoothing(items)
        if causal_info["applied"]:
            name += "_causal"

    final_raw_dir = SUBMISSIONS / "exp055" / f"{name}_raw"
    if final_raw_dir != raw_dir:
        final_raw_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(final_raw_dir, items)

    zip_path = SUBMISSIONS / f"exp055_{name}.zip"
    create_submission_zip(final_raw_dir, zip_path, filenames)

    entry_summary = {
        "experiment": "exp055",
        "scheme": name,
        "manifest_sources": list(manifest.keys()),
        "weights": weights,
        "causal_smoothing": causal_info,
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "files": len(filenames),
        "recommended_weights_source": str(RECOMMENDED),
    }
    summary_path = ANALYSIS_DIR / f"analysis_summary_{name}.json"
    summary_path.write_text(json.dumps(entry_summary, indent=2), encoding="utf-8")
    print(f"wrote manifest: {summary_path}", flush=True)
    print(f"wrote submission zip: {zip_path}", flush=True)


if __name__ == "__main__":
    main()
