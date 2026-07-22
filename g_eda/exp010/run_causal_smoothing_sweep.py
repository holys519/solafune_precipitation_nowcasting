#!/usr/bin/env python3
"""Causal-only temporal-smoothing OOF sweep for the green champion (g_eda/exp010).

Motivation (see doc/submission_registry.md, 2026-07-20 organizer ruling + doc/public_scores.md):
exp046 shipped causal-only temporal smoothing (center=0.85/prev=0.15/next=0) on top of exp038,
and it beat exp038 solo on the public LB (-0.00025). But those weights were never actually OOF
tuned for the causal-only (2-tap) case -- they are exp036/037's OLD bidirectional weights with
next_weight's share folded into prev_weight after the fact. This script re-tunes causal-only
smoothing properly on OOF predictions for the CURRENT green champion, exp038_sigmafixed, and
(optionally, if a 5-fold cache is available) exp047.

Follows the two-stage pattern of g_eda/exp004 (run_temporal_smoothing.py + run_joint_postprocess.py),
adapted to be causal-only (no next_weight tap) and to add a second causal tap (prev2, T-60min):

  Stage 1: search (center_weight, prev_weight) with prev2_weight=0 (2-tap causal), then extend to
           (center_weight, prev_weight, prev2_weight) (3-tap causal), next_weight fixed at 0.
  Stage 2: given the winning smoothing weights, jointly re-sweep blur sigma x per-satellite
           value_threshold on top (mirrors g_eda/exp004/run_joint_postprocess.py's stage 2).

Source OOF prediction caches follow the g_eda/exp003 caching convention
(outputs/g_eda/exp003/<exp>_oof_pred.npz, built by g_eda/exp003/run_blend_curve.py --cache).
This script only reads those caches -- it does not modify g_eda/exp003, g_eda/exp004,
g_experiments/exp038, g_experiments/exp046, or g_experiments/exp047.

CPU-only. Writes recommended_causal_weights.json + CAUSAL_SMOOTHING.md under g_eda/exp010/,
and a raw sweep CSV under outputs/g_eda/exp010/.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXP010_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp003"
OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp010"
TRAIN_CSV = PROJECT_DIR / "data" / "train_dataset" / "train_dataset.csv"
SATELLITES = ("goes", "himawari", "meteosat")

# exp046's shipped (untuned) causal weights -- the baseline this sweep must beat.
EXP046_BASELINE = {"center_weight": 0.85, "prev_weight": 0.15, "prev2_weight": 0.0, "next_weight": 0.0}
MAX_GAP_MINUTES = 30

BLUR_GRID = (0.0, 0.5, 0.75, 1.0, 1.25)
THRESHOLD_GRID = (0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25)

sys.path.insert(0, str(EXP010_DIR))
from causal_smoothing import CausalSmoothingConfigError, apply_temporal_smoothing  # noqa: E402


def _self_check_guard() -> None:
    """Prove the causal_only compliance guard actually fires before trusting the sweep."""
    items = [
        {"name_location": "x", "datetime": "2023-01-01 00:00:00", "array": np.ones((2, 2), dtype=np.float32)},
        {"name_location": "x", "datetime": "2023-01-01 00:30:00", "array": np.ones((2, 2), dtype=np.float32) * 2},
    ]
    bad_cfg = {"temporal_smoothing": {"enabled": True, "center_weight": 0.8, "prev_weight": 0.1,
                                       "next_weight": 0.1, "causal_only": True}}
    try:
        apply_temporal_smoothing(items, bad_cfg)
    except CausalSmoothingConfigError:
        pass
    else:
        raise AssertionError("causal_only guard did not fire for next_weight > 0 -- compliance bug")
    ok_cfg = {"temporal_smoothing": {"enabled": True, "center_weight": 0.85, "prev_weight": 0.15,
                                      "next_weight": 0.0, "causal_only": True}}
    apply_temporal_smoothing([dict(it) for it in items], ok_cfg)  # must not raise
    print("causal_only guard self-check: OK (raises on next_weight>0, passes on next_weight=0)",
          flush=True)


def gaussian_blur(pred: np.ndarray, sigma: float) -> np.ndarray:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(pred, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(pred)
    for i, k in enumerate(kernel):
        out += k * padded[:, i : i + pred.shape[1], :]
    padded = np.pad(out, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(pred)
    for i, k in enumerate(kernel):
        out += k * padded[:, :, i : i + pred.shape[2]]
    return out


def tile_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(pred - target).reshape(pred.shape[0], -1).mean(axis=1))


def finite_min(rows: list, key):
    """min() by `key`, treating NaN as +inf so a degenerate (NaN) candidate can never win just
    because it happens to be first (Python's NaN comparisons are always False, so plain min()
    silently keeps whichever element it saw first once a NaN is involved -- see smooth()'s
    docstring). Raises if every candidate is non-finite."""
    def safe_key(row):
        value = key(row)
        return value if math.isfinite(value) else math.inf
    best = min(rows, key=safe_key)
    if not math.isfinite(key(best)):
        raise ValueError("finite_min: every candidate had a non-finite score")
    return best


def load_meta() -> dict[str, tuple[str, datetime]]:
    meta: dict[str, tuple[str, datetime]] = {}
    with TRAIN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            meta[row["unique_id"]] = (row["name_location"], datetime.fromisoformat(row["datetime"]))
    return meta


def build_causal_neighbor_indices(
    unique_id: np.ndarray, meta: dict[str, tuple[str, datetime]], max_gap_minutes: int
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorization-friendly version of causal_smoothing._causal_neighbors: for each tile,
    find the index of its T-30min neighbor (prev) and, if that exists and is itself T-30-from-a-
    T-60 neighbor, the T-60min neighbor (prev2). Same location, same chained-causal semantics as
    the reference apply_temporal_smoothing in causal_smoothing.py."""
    n = len(unique_id)
    by_location: dict[str, list[int]] = {}
    for i, uid in enumerate(unique_id):
        loc, _ = meta[uid]
        by_location.setdefault(loc, []).append(i)

    prev_idx = np.full(n, -1, dtype=np.int64)
    prev2_idx = np.full(n, -1, dtype=np.int64)
    for idxs in by_location.values():
        idxs_sorted = sorted(idxs, key=lambda i: meta[unique_id[i]][1])
        times = [meta[unique_id[i]][1] for i in idxs_sorted]
        for pos, i in enumerate(idxs_sorted):
            if pos > 0:
                gap1 = (times[pos] - times[pos - 1]).total_seconds() / 60.0
                if 0 < gap1 <= max_gap_minutes:
                    prev_idx[i] = idxs_sorted[pos - 1]
                    if pos > 1:
                        gap2 = (times[pos - 1] - times[pos - 2]).total_seconds() / 60.0
                        if 0 < gap2 <= max_gap_minutes:
                            prev2_idx[i] = idxs_sorted[pos - 2]
    return prev_idx, prev2_idx


def smooth(pred: np.ndarray, cw: float, pw: float, p2w: float,
           prev_idx: np.ndarray, prev2_idx: np.ndarray) -> np.ndarray:
    """Vectorized causal smoothing matching causal_smoothing.apply_temporal_smoothing exactly:
    next_weight is always 0 here (causal-only by construction), renormalizes by weight actually
    used per-tile (tiles without a prev/prev2 fall back toward center-weight-only)."""
    has_prev = prev_idx >= 0
    has_prev2 = prev2_idx >= 0
    weighted = cw * pred
    total = np.full(pred.shape[0], cw, dtype=np.float32)
    if pw > 0:
        weighted[has_prev] += pw * pred[prev_idx[has_prev]]
        total[has_prev] += pw
    if p2w > 0:
        weighted[has_prev2] += p2w * pred[prev2_idx[has_prev2]]
        total[has_prev2] += p2w
    # Guard against exact-zero total weight (e.g. cw=0 and no causal neighbor available for a
    # given tile) -- matches causal_smoothing.apply_temporal_smoothing's max(total_weight, 1e-8).
    # Without this, a handful of NaN tiles at grid-edge weight combos silently poison every
    # downstream min()-based "best" selection (NaN comparisons are always False in Python, so a
    # NaN-valued row that appears first in a list is never displaced by a real-valued row).
    total = np.maximum(total, 1e-8)
    return weighted / total[:, None, None]


def load_source(exp_name: str) -> dict[str, np.ndarray] | None:
    path = CACHE_DIR / f"{exp_name}_oof_pred.npz"
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=False)
    order = np.argsort(d["unique_id"])
    return {
        "pred": d["pred"].astype(np.float32)[order],
        "target": d["target"].astype(np.float32)[order],
        "unique_id": d["unique_id"][order],
        "satellite": d["satellite"][order],
    }


def score(pred: np.ndarray, target: np.ndarray, sat_masks: dict[str, np.ndarray]) -> dict[str, float]:
    per_tile = tile_rmse(pred, target)
    result = {"overall": float(per_tile.mean())}
    for sat, mask in sat_masks.items():
        result[sat] = float(per_tile[mask].mean()) if mask.any() else float("nan")
    return result


def run_for_source(exp_name: str, data: dict[str, np.ndarray]) -> dict[str, Any]:
    pred = data["pred"]
    target = data["target"]
    unique_id = data["unique_id"]
    satellite = data["satellite"]
    sat_masks = {sat: satellite == sat for sat in SATELLITES}

    meta = load_meta()
    prev_idx, prev2_idx = build_causal_neighbor_indices(unique_id, meta, MAX_GAP_MINUTES)
    n_prev = int((prev_idx >= 0).sum())
    n_prev2 = int((prev2_idx >= 0).sum())
    print(f"[{exp_name}] tiles={len(unique_id)} with prev={n_prev} ({n_prev / len(unique_id):.1%}) "
          f"prev2={n_prev2} ({n_prev2 / len(unique_id):.1%})", flush=True)

    report: list[str] = []
    csv_rows: list[dict[str, Any]] = []

    # --- (a) no smoothing
    no_smooth_score = score(pred, target, sat_masks)

    # --- (b) exp046 shipped (untuned) causal weights, applied causally
    b = EXP046_BASELINE
    exp046_pred = smooth(pred, b["center_weight"], b["prev_weight"], b["prev2_weight"], prev_idx, prev2_idx)
    exp046_score = score(exp046_pred, target, sat_masks)

    # --- (c) 2-tap causal OOF sweep: center_weight + prev_weight = 1, prev2_weight = 0
    two_tap_rows = []
    for cw100 in range(0, 101):
        cw = cw100 / 100.0
        pw = 1.0 - cw
        pred_smoothed = smooth(pred, cw, pw, 0.0, prev_idx, prev2_idx)
        s = score(pred_smoothed, target, sat_masks)
        two_tap_rows.append({"center_weight": cw, "prev_weight": pw, "prev2_weight": 0.0, **s})
        csv_rows.append({"stage": "2tap", **two_tap_rows[-1]})
    best_2tap = finite_min(two_tap_rows, key=lambda r: r["overall"])

    # --- (d) 3-tap causal OOF sweep: center + prev + prev2 = 1, simplex grid step 0.05
    three_tap_rows = []
    step = 5  # percent
    for cw100 in range(0, 101, step):
        for pw100 in range(0, 101 - cw100, step):
            p2w100 = 100 - cw100 - pw100
            cw, pw, p2w = cw100 / 100.0, pw100 / 100.0, p2w100 / 100.0
            pred_smoothed = smooth(pred, cw, pw, p2w, prev_idx, prev2_idx)
            s = score(pred_smoothed, target, sat_masks)
            three_tap_rows.append({"center_weight": cw, "prev_weight": pw, "prev2_weight": p2w, **s})
            csv_rows.append({"stage": "3tap", **three_tap_rows[-1]})
    best_3tap = finite_min(three_tap_rows, key=lambda r: r["overall"])

    use_3tap = best_3tap["overall"] < best_2tap["overall"] - 1e-6
    winning_smooth = best_3tap if use_3tap else best_2tap
    winning_pred = smooth(pred, winning_smooth["center_weight"], winning_smooth["prev_weight"],
                           winning_smooth["prev2_weight"], prev_idx, prev2_idx)

    # --- stage 2: joint blur sigma x per-satellite value_threshold re-sweep on the winning smooth
    best_joint = None
    for sigma in BLUR_GRID:
        blurred = winning_pred if sigma == 0.0 else gaussian_blur(winning_pred, sigma)
        per_tile_by_thr = {thr: tile_rmse(np.where(blurred < thr, 0.0, blurred), target)
                           for thr in THRESHOLD_GRID}
        sat_thr = {}
        total = 0.0
        for sat, mask in sat_masks.items():
            if not mask.any():
                sat_thr[sat] = 0.0
                continue
            best_thr = min(THRESHOLD_GRID, key=lambda thr: float(per_tile_by_thr[thr][mask].mean()))
            sat_thr[sat] = best_thr
            total += float(per_tile_by_thr[best_thr][mask].mean()) * mask.mean()
        candidate = {"sigma": sigma, "thresholds": sat_thr, "overall": total}
        if not math.isfinite(candidate["overall"]):
            continue
        if best_joint is None or candidate["overall"] < best_joint["overall"]:
            best_joint = candidate
    if best_joint is None:
        raise ValueError("blur/threshold joint sweep: every candidate had a non-finite score")

    final_blurred = winning_pred if best_joint["sigma"] == 0.0 else gaussian_blur(winning_pred, best_joint["sigma"])
    final_pred = final_blurred.copy()
    for sat, mask in sat_masks.items():
        thr = best_joint["thresholds"][sat]
        if thr > 0:
            final_pred[mask] = np.where(final_pred[mask] < thr, 0.0, final_pred[mask])
    final_score = score(final_pred, target, sat_masks)

    report += [
        f"## Source: {exp_name} (n={len(unique_id)} OOF tiles)",
        "",
        f"- (a) no smoothing:                        {no_smooth_score['overall']:.5f}",
        f"- (b) exp046 shipped (untuned, 0.85/0.15/0.0): {exp046_score['overall']:.5f} "
        f"(delta vs a: {exp046_score['overall'] - no_smooth_score['overall']:+.5f})",
        f"- (c) 2-tap OOF-tuned (center={best_2tap['center_weight']:.2f}, "
        f"prev={best_2tap['prev_weight']:.2f}): {best_2tap['overall']:.5f} "
        f"(delta vs b: {best_2tap['overall'] - exp046_score['overall']:+.5f})",
        f"- (d) 3-tap OOF-tuned (center={best_3tap['center_weight']:.2f}, "
        f"prev={best_3tap['prev_weight']:.2f}, prev2={best_3tap['prev2_weight']:.2f}): "
        f"{best_3tap['overall']:.5f} (delta vs c: {best_3tap['overall'] - best_2tap['overall']:+.5f}) "
        f"-> {'ADOPTED' if use_3tap else 'not adopted (no improvement over 2-tap)'}",
        f"- (e) + joint blur/threshold re-opt (sigma={best_joint['sigma']}, "
        f"thresholds={best_joint['thresholds']}): {final_score['overall']:.5f} "
        f"(delta vs {'d' if use_3tap else 'c'}: "
        f"{final_score['overall'] - winning_smooth['overall']:+.5f})",
        "",
        f"Per-satellite (final stack e): "
        + ", ".join(f"{sat}={final_score[sat]:.5f}" for sat in SATELLITES),
        "",
        f"**Total delta (a -> e): {final_score['overall'] - no_smooth_score['overall']:+.5f}**",
        f"**Delta vs exp046 shipped (b -> e): {final_score['overall'] - exp046_score['overall']:+.5f}**",
        "",
    ]

    recommendation = {
        "schema_version": 1,
        "source_experiment": exp_name,
        "generated_by": "g_eda/exp010/run_causal_smoothing_sweep.py",
        "compliance": "causal_only (2026-07-20 ruling): next_weight is always 0 in this recommendation",
        "temporal_smoothing": {
            "enabled": True,
            "causal_only": True,
            "center_weight": winning_smooth["center_weight"],
            "prev_weight": winning_smooth["prev_weight"],
            "prev2_weight": winning_smooth["prev2_weight"],
            "next_weight": 0.0,
            "max_gap_minutes": MAX_GAP_MINUTES,
        },
        "blur_sigma": best_joint["sigma"],
        "per_satellite_value_threshold": best_joint["thresholds"],
        "oof_scores": {
            "no_smoothing": no_smooth_score["overall"],
            "exp046_shipped_baseline": exp046_score["overall"],
            "tuned_2tap": best_2tap["overall"],
            "tuned_3tap": best_3tap["overall"],
            "final_with_joint_postprocess": final_score["overall"],
        },
        "used_3tap": use_3tap,
    }
    return {"report_lines": report, "csv_rows": csv_rows, "recommendation": recommendation}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _self_check_guard()

    sources_to_try = ["exp038_sigmafixed", "exp047"]
    results: dict[str, Any] = {}
    skipped: list[str] = []
    for exp_name in sources_to_try:
        data = load_source(exp_name)
        if data is None:
            skipped.append(exp_name)
            continue
        print(f"running sweep for {exp_name} ...", flush=True)
        results[exp_name] = run_for_source(exp_name, data)

    if not results:
        raise FileNotFoundError(
            "No OOF caches found under outputs/g_eda/exp003/*_oof_pred.npz for "
            f"{sources_to_try}. Build the exp038_sigmafixed cache first, e.g.:\n"
            "  cd g_eda/exp003 && sbatch singularity_cache_exp038.sh exp038_sigmafixed exp038 exp038_sigmafixed"
        )

    all_csv_rows: list[dict[str, Any]] = []
    report_lines = ["# Causal-only temporal smoothing OOF sweep (g_eda/exp010)", "",
                     "Re-tunes exp046's causal-only temporal smoothing (which shipped with untuned",
                     "weights copied from the old bidirectional design) on OOF predictions.",
                     "Stack: raw prediction -> causal smoothing (T, T-30, T-60 only, next_weight=0)",
                     "-> blur sigma -> per-satellite value_threshold. All post-processing is fit on",
                     "OOF and next_weight is asserted 0 by causal_smoothing.py's causal_only guard.",
                     ""]
    primary = "exp038_sigmafixed" if "exp038_sigmafixed" in results else next(iter(results))
    for exp_name, res in results.items():
        report_lines += res["report_lines"]
        all_csv_rows += [{"source": exp_name, **row} for row in res["csv_rows"]]

    if skipped:
        report_lines += ["## Skipped sources", ""]
        for exp_name in skipped:
            report_lines.append(
                f"- {exp_name}: no OOF cache at outputs/g_eda/exp003/{exp_name}_oof_pred.npz yet "
                "(follow-up once its 5-fold checkpoints/cache are available)."
            )
        report_lines.append("")

    report_lines += [
        "## Recommended config (primary source: " + primary + ")",
        "",
        "See `recommended_causal_weights.json` (schema documented in its own `schema_version` "
        "field / this report) for the exact machine-readable recommendation consumed downstream "
        "(e.g. by a future exp055 harvest build).",
        "",
        "```json",
        json.dumps(results[primary]["recommendation"], indent=2),
        "```",
    ]

    csv_path = OUT_DIR / "causal_smoothing_sweep.csv"
    if all_csv_rows:
        fieldnames = list(all_csv_rows[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_csv_rows)

    recommended_path = EXP010_DIR / "recommended_causal_weights.json"
    recommended_path.write_text(json.dumps(results[primary]["recommendation"], indent=2) + "\n",
                                 encoding="utf-8")

    report_path = EXP010_DIR / "CAUSAL_SMOOTHING.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines), flush=True)
    print(f"\nwrote {recommended_path}", flush=True)
    print(f"wrote {report_path}", flush=True)
    print(f"wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
