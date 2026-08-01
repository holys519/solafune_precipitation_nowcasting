#!/usr/bin/env python3
"""g_experiments/exp065: architecture-diverse champion ensemble submission builder.

2026-07-30 context: `exp064_effb3` (solo LB 0.6720097338314985) is the new green champion,
overturning fold0/4's "mixed" verdict by a wide margin -- see `doc/public_scores.md` 2026-07-30
追記5 and `doc/plan/round8_campaign_2026-07-27.md`'s "2026-07-30 update". The round8 plan's item
(b) priority follow-up is to ensemble the new champion with an architecturally-DIFFERENT model
(not just another EfficientNet-family seed), on the theory that error decorrelation across genuinely
different feature extractors is more likely to generalize than another same-family variant --
exactly the `exp056_seed_ens_*` lesson (same-architecture seed averaging plateaued at 2 seeds and
scaling to 4/6 seeds made things worse, see `doc/public_scores.md` 2026-07-27 entries) but applied
along the *architecture* axis instead of the *seed* axis.

Sources (registry_guard-verified green, see g_eda/exp011/sources.json / green_allowlist.json):
  - exp056             -- from-scratch CompactUNet (the original champion architecture, seed42)
  - exp064_effb3       -- timm efficientnet_b3 pretrained encoder (NEW solo champion)
  - exp064_effv2s      -- timm tf_efficientnetv2_s pretrained encoder (2nd-best solo)
  - exp064_swin_lr2e4  -- timm swin_tiny_patch4_window7_224 pretrained encoder (transformer,
                          architecturally the most distinct member, best fold0/4 gate result since
                          exp056)

Unlike g_experiments/exp055 (which read g_eda/exp011/recommended_weights.json, an IN-SAMPLE-fit
weight computed over the entire sources.json manifest), this script reads
g_eda/exp011/nested_blend.py's `in_sample_weights` field out of `nested_blend_report.json` --
that field is fit on the SAME full data as optimize_blend.py would produce, but ONLY over the
--sources actually requested here (this experiment's 4-member subset, not sources.json's full
6-entry manifest, which still also carries the older exp038_sigmafixed/exp040_metric sources this
experiment does not want mixed in). The nested_blend_report.json's nested_score/overfitting_gap is
what tells you whether those in_sample_weights are trustworthy -- read NESTED_BLEND.md before
trusting this script's output, and run l_eda/exp005/submission_gate.py on the emitted
oof_sample_metrics.csv (nested_blend.py already writes one) before spending an actual LB submission
on the zip this script produces.

Compliance (same two properties exp055 established, both re-enforced independently here):
  1. Every source is re-asserted green via registry_guard.assert_green at build time.
  2. No overlap-patch step anywhere in this file -- overlap patch is permanently disqualifying
     (2026-07-20 ruling). Post-processing is blend -> causal-only smoothing (only if
     g_eda/exp010 has published a recommendation; next_weight must be exactly 0, enforced twice
     as in exp055) -> done.

Usage:
    python3 build_submission.py --dry-run
    python3 build_submission.py
    python3 build_submission.py --sources exp056 exp064_effb3 exp064_swin_lr2e4   # 3-way instead
    python3 build_submission.py --skip-causal-smoothing
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
ANALYSIS_DIR = ROOT / "outputs" / "analysis" / "exp065"
EVALUATION_CSV = ROOT / "data" / "evaluation_dataset" / "evaluation_target.csv"
NESTED_REPORT = ROOT / "outputs" / "g_eda" / "exp011" / "nested_blend_report.json"
MANIFEST = EXP011 / "sources.json"
# 2026-07-30: default was exp010's ORIGINAL recommendation, OOF-tuned against exp038_sigmafixed
# solo (a different model's error structure). Once exp010's sweep was re-run against this
# ensemble's own OOF (--sources champion_ensemble_nested_blend --cache-dir ../../outputs/g_eda/
# exp011), recommended_causal_weights_exp065.json became the properly-tuned-for-THIS-ensemble
# file and is now the default; override with --causal-json to reproduce the old behavior.
CAUSAL_JSON = EXP010 / "recommended_causal_weights_exp065.json"

DEFAULT_SOURCES = ["exp056", "exp064_effb3", "exp064_effv2s", "exp064_swin_lr2e4"]

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
        raise FileNotFoundError(
            f"missing eval prediction directory: {source_dir} -- has this source's "
            "inference.py/make_submission.py been run yet?"
        )
    return {path.name: path for path in source_dir.glob("*.tif")}


def load_weights(names: list[str]) -> dict[str, float]:
    if not NESTED_REPORT.exists():
        raise FileNotFoundError(
            f"{NESTED_REPORT} not found -- run g_eda/exp011/nested_blend.py --sources "
            f"{' '.join(names)} first (after caching each source with "
            "`optimize_blend.py --cache NAME`)."
        )
    rec = json.loads(NESTED_REPORT.read_text(encoding="utf-8"))
    report_names = set(rec["names"])
    if report_names != set(names):
        raise ValueError(
            f"nested_blend_report.json was computed for {sorted(report_names)} but this build "
            f"was asked for {sorted(names)} -- re-run nested_blend.py --sources {' '.join(names)} "
            "first (it overwrites the report for whichever --sources were passed)."
        )
    weights = {k: float(v) for k, v in rec["in_sample_weights"].items()}
    total = sum(weights.values())
    if not 0.999 <= total <= 1.001:
        raise ValueError(f"weights must sum to 1, got {total}: {weights}")

    gap = float(rec["overfitting_gap"])
    nested_vs_solo = float(rec["nested_vs_best_solo"])
    print(
        f"loaded in_sample_weights={weights} from {NESTED_REPORT}\n"
        f"  in_sample_score={rec['in_sample_score']:.5f}  nested_score={rec['nested_score']:.5f}  "
        f"overfitting_gap={gap:+.5f}\n"
        f"  nested_vs_best_solo={nested_vs_solo:+.5f} (best solo: {rec['best_solo_name']} "
        f"{rec['best_solo_score']:.5f})",
        flush=True,
    )
    if nested_vs_solo >= 0:
        print(
            "WARNING: the HONEST nested-CV score does not beat the best solo member -- the "
            "in-sample weights below may not generalize (see doc/public_scores.md's 2026-07-22 "
            "exp055 postmortem for exactly this failure mode). Do not spend an LB submission on "
            "this zip without also checking l_eda/exp005/submission_gate.py's verdict.",
            flush=True,
        )
    return weights


def assert_sources_green(names: list[str]) -> None:
    registry_guard.assert_all_green(names)


def blend_evaluation(
    name: str, weights: dict[str, float], rows: list[dict[str, str]], files: dict[str, dict[str, Path]]
) -> tuple[Path, list[tuple[dict, np.ndarray, Path]]]:
    raw_dir = SUBMISSIONS / "exp065" / f"{name}_raw"
    destination = raw_dir / "test_files"
    destination.mkdir(parents=True, exist_ok=True)

    blended_items: list[tuple[dict, np.ndarray, Path]] = []
    for index, row in enumerate(rows, start=1):
        filename = row["gpm_imerg_filename"]
        blended = None
        template = None
        for source_name, weight in weights.items():
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


def apply_causal_smoothing(
    items: list[tuple[dict, np.ndarray, Path]],
) -> tuple[list[tuple[dict, np.ndarray, Path]], dict]:
    """Identical contract to g_experiments/exp055's version (see that file's docstring): loads
    g_eda/exp010's recommendation if present, hard-refuses anything non-causal (next_weight must
    be exactly 0), applies it plus the accompanying blur/threshold post-process."""
    if not CAUSAL_JSON.exists():
        return items, {
            "applied": False,
            "reason": "g_eda/exp010/recommended_causal_weights.json does not exist yet -- causal-only "
            "smoothing hook is wired but inactive until that experiment publishes a recommendation.",
        }

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

    smoothing_items = [
        {"name_location": row["name_location"], "datetime": row["datetime"], "array": array}
        for row, array, _ in items
    ]
    smoothed = apply_temporal_smoothing(smoothing_items, {"temporal_smoothing": smoothing_cfg})
    new_items = [
        (row, smoothed_item["array"], template)
        for (row, _, template), smoothed_item in zip(items, smoothed)
    ]

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

    return new_items, {
        "applied": True,
        "source_experiment": rec.get("source_experiment"),
        "temporal_smoothing": smoothing_cfg,
        "blur_sigma": blur_sigma,
        "per_satellite_value_threshold": thresholds,
    }


def gaussian_blur_2d(array: np.ndarray, sigma: float) -> np.ndarray:
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
    global CAUSAL_JSON
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    parser.add_argument("--name", default="champion_ensemble")
    parser.add_argument("--causal-json", type=Path, default=None,
                         help=f"override the causal-smoothing recommendation file "
                         f"(default: {CAUSAL_JSON})")
    parser.add_argument("--skip-causal-smoothing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.causal_json is not None:
        CAUSAL_JSON = args.causal_json

    names = args.sources
    manifest = load_manifest()
    missing = [n for n in names if n not in manifest]
    if missing:
        raise KeyError(f"{missing} not found in {MANIFEST} -- add manifest entries first")

    # Hard, code-enforced refusal -- not just a comment/convention.
    assert_sources_green(names)

    rows = read_evaluation_rows()
    filenames = [row["gpm_imerg_filename"] for row in rows]
    files = {}
    for source_name in names:
        pred_dir = ROOT / manifest[source_name]["eval_pred_dir"]
        files[source_name] = source_files(pred_dir)
        missing_files = set(filenames) - set(files[source_name])
        if missing_files:
            raise ValueError(f"{source_name}: {len(missing_files)} evaluation files missing under {pred_dir}")

    weights = load_weights(names)
    name = args.name

    print(json.dumps({"scheme": name, "weights": weights, "sources": names, "files": len(filenames)}, indent=2), flush=True)
    if args.dry_run:
        return

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir, items = blend_evaluation(name, weights, rows, files)

    causal_info = {"applied": False, "reason": "--skip-causal-smoothing passed"}
    if not args.skip_causal_smoothing:
        items, causal_info = apply_causal_smoothing(items)
        if causal_info["applied"]:
            name += "_causal"

    final_raw_dir = SUBMISSIONS / "exp065" / f"{name}_raw"
    if final_raw_dir != raw_dir:
        final_raw_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(final_raw_dir, items)

    zip_path = SUBMISSIONS / f"exp065_{name}.zip"
    create_submission_zip(final_raw_dir, zip_path, filenames)

    entry_summary = {
        "experiment": "exp065",
        "scheme": name,
        "sources": names,
        "weights": weights,
        "causal_smoothing": causal_info,
        "zip": str(zip_path),
        "zip_sha256": sha256(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "files": len(filenames),
        "nested_blend_report": str(NESTED_REPORT),
    }
    summary_path = ANALYSIS_DIR / f"analysis_summary_{name}.json"
    summary_path.write_text(json.dumps(entry_summary, indent=2), encoding="utf-8")
    print(f"wrote manifest: {summary_path}", flush=True)
    print(f"wrote submission zip: {zip_path}", flush=True)
    print(
        "REMINDER: check l_eda/exp005/submission_gate.py's verdict for this candidate before "
        "spending an actual Kaggle submission slot on the zip above.",
        flush=True,
    )


if __name__ == "__main__":
    main()
