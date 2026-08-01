#!/usr/bin/env python3
"""g_eda/exp011: outer-cross-fit (nested) blend weight search.

`optimize_blend.py`'s `search_two_way`/`search_three_way`/`search_greedy` fit the blend weight
by grid search directly against the *same* OOF tiles used to report the gain -- no outer
train/holdout split. `doc/public_scores.md` (2026-07-22, exp055 postmortem) diagnosed this
explicitly as the cause of a severe OOF/LB inversion: an in-sample-fit 48/52
exp038_sigmafixed x exp040_metric blend looked like -0.0087 OOF gain and came back +0.0006 worse
on the public LB. The doc's own conclusion: "must be redone with an outer cross-fit (fit weights
on a subset of folds/locations, score on the held-out remainder, repeated) rather than a single
in-sample fit on the full OOF."

This script is that redo. It reuses this project's existing 5-fold GroupKFold split as the outer
CV unit (each fold's tiles come from locations the fold-holdout model never trained on, so it is
already a natural, leakage-free split for this purpose -- no new fold assignment needed):

    for each outer fold k:
        fit the blend weight using OOF tiles from folds != k only
        evaluate that weight on fold k's tiles (which the fit never saw)

The concatenation of all 5 held-out evaluations is a genuine nested-CV blend score, directly
comparable to (and, if this project's diagnosis is right, *much* closer to LB than) the naive
in-sample score `optimize_blend.py --analyze` reports. The gap between the two IS the
overfitting `doc/public_scores.md` flagged -- this script reports it explicitly rather than
requiring another failed submission to notice it next time.

Implements N in {1, 2, 3} exactly (grid/simplex search) and N > 3 via greedy forward blend-in
(2026-07-30, added for the exp056 / exp064_effb3 / exp064_effv2s / exp064_swin_lr2e4 4-source
architecture-diverse ensemble -- see `fit_weight()` below, ported from `optimize_blend.py`'s
`search_greedy` global-path branch). Global weight only, no per-satellite split -- with as few as
2 locations in outer fold 0, splitting further by satellite would leave single-digit tile counts
per cell.

Usage:
    python3 nested_blend.py                                # uses sources.json, same manifest as optimize_blend.py
    python3 nested_blend.py --sources exp038_sigmafixed exp040_metric

Outputs outputs/g_eda/exp011/nested_blend_report.json + NESTED_BLEND.md, and
outputs/analysis/{first_source}_nested_blend/oof_sample_metrics.csv in the same schema
`analyze_oof.py` uses elsewhere in this repo, so `l_eda/exp005`'s bootstrap_ci.py /
leakage_audit.py can be pointed at the nested blend directly, e.g.:

    python3 l_eda/exp005/submission_gate.py --baseline exp038_sigmafixed --candidate <name>_nested_blend
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp011"
ANALYSIS_DIR = PROJECT_DIR / "outputs" / "analysis"

sys.path.insert(0, str(EXP_DIR))
import registry_guard  # noqa: E402
import optimize_blend as ob  # noqa: E402


def load_aligned_with_fold(manifest: list[dict]):
    """Same alignment as optimize_blend.load_aligned, plus the per-tile fold id and location.

    Duplicated (not refactored into optimize_blend.py) deliberately: optimize_blend.py's
    load_aligned is a stable 4-tuple return already consumed by g_experiments/exp055 and the
    sbatch scripts; this keeps that surface untouched.
    """
    names = [entry["name"] for entry in manifest]
    registry_guard.assert_all_green(names)

    caches = {}
    for entry in manifest:
        path = OUT_DIR / f"{entry['name']}_oof_pred.npz"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing -- run optimize_blend.py --cache {entry['name']} first")
        caches[entry["name"]] = np.load(path, allow_pickle=False)

    ref_name = names[0]
    ref_ids = caches[ref_name]["unique_id"]
    for name in names[1:]:
        if not np.array_equal(np.sort(caches[name]["unique_id"]), np.sort(ref_ids)):
            raise ValueError(f"{name}: unique_id set differs from {ref_name} -- caches are not comparable")
    order = {name: np.argsort(caches[name]["unique_id"]) for name in names}

    aligned = {name: caches[name]["pred"].astype(np.float32)[order[name]] for name in names}
    target = caches[ref_name]["target"].astype(np.float32)[order[ref_name]]
    satellite = caches[ref_name]["satellite"][order[ref_name]]
    unique_id = caches[ref_name]["unique_id"][order[ref_name]]
    fold = caches[ref_name]["fold"][order[ref_name]]
    return aligned, target, satellite, unique_id, fold


def unique_id_to_location(exp_name_for_locations: str) -> dict[str, str]:
    """Look up name_location per unique_id from an existing analyze_oof.py cache.

    Any green source's oof_sample_metrics.csv carries the same unique_id -> name_location
    mapping (it's a property of the row, not the model), so the first manifest source works.
    """
    path = ANALYSIS_DIR / exp_name_for_locations / "oof_sample_metrics.csv"
    with path.open(newline="") as f:
        return {row["unique_id"]: row["name_location"] for row in csv.DictReader(f)}


def fit_weight(names: list[str], aligned_subset: dict[str, np.ndarray], target_subset: np.ndarray) -> dict[str, float]:
    """Grid-search the global blend weight on one (training) subset only. No I/O side effects."""
    if len(names) == 1:
        return {names[0]: 1.0}
    if len(names) == 2:
        a, b = names
        best_w, best_val = 0.0, float(ob.tile_rmse(aligned_subset[a], target_subset).mean())
        for w in np.round(np.arange(0.0, 1.0001, 0.01), 2):
            pred = (1.0 - w) * aligned_subset[a] + w * aligned_subset[b]
            val = float(ob.tile_rmse(pred, target_subset).mean())
            if val < best_val:
                best_val, best_w = val, w
        return {a: round(1.0 - best_w, 4), b: round(best_w, 4)}
    if len(names) == 3:
        step = 0.05
        steps = int(round(1.0 / step))
        best_weights, best_val = {names[0]: 1.0, names[1]: 0.0, names[2]: 0.0}, float(
            ob.tile_rmse(aligned_subset[names[0]], target_subset).mean()
        )
        for i in range(steps + 1):
            for j in range(steps + 1 - i):
                w0, w1 = i * step, j * step
                w2 = 1.0 - w0 - w1
                pred = w0 * aligned_subset[names[0]] + w1 * aligned_subset[names[1]] + w2 * aligned_subset[names[2]]
                val = float(ob.tile_rmse(pred, target_subset).mean())
                if val < best_val:
                    best_val = val
                    best_weights = {names[0]: round(w0, 2), names[1]: round(w1, 2), names[2]: round(w2, 2)}
        return best_weights
    # N > 3: greedy forward blend-in, restricted to this call's subset only (this function is
    # always called with either the full OOF or one outer-fold's TRAIN-only subset -- see
    # nested_cross_fit below -- so "restricted to this subset" is what keeps the outer fold honest;
    # the caller must never pass the held-out fold's tiles in here). Global weight only, no
    # per-satellite split, consistent with this file's 2-way/3-way branches above (mirrors
    # optimize_blend.py's search_greedy global-path branch, lines ~376-397, without the
    # per-satellite variant that function also computes -- that variant isn't meaningful here since
    # each outer fold's 2-location subset would leave single-digit per-satellite tile counts).
    step = 0.05
    solo_scores = {n: float(ob.tile_rmse(aligned_subset[n], target_subset).mean()) for n in names}
    remaining = set(names)
    first = min(remaining, key=lambda n: solo_scores[n])
    remaining.remove(first)
    weights = {first: 1.0}
    current_pred = aligned_subset[first].copy()
    while remaining:
        best = None  # (candidate, w_new, value, pred)
        for candidate in remaining:
            for w_new in np.round(np.arange(0.0, 1.0001, step), 2):
                pred = (1.0 - w_new) * current_pred + w_new * aligned_subset[candidate]
                value = float(ob.tile_rmse(pred, target_subset).mean())
                if best is None or value < best[2]:
                    best = (candidate, w_new, value, pred)
        chosen, w_new, _, pred = best
        for name in weights:
            weights[name] *= (1.0 - w_new)
        weights[chosen] = weights.get(chosen, 0.0) + w_new
        current_pred = pred
        remaining.remove(chosen)
    return {n: round(w, 4) for n, w in weights.items()}


def blend_pred(names: list[str], aligned_subset: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    out = np.zeros_like(aligned_subset[names[0]])
    for name in names:
        out += weights[name] * aligned_subset[name]
    return out


def nested_cross_fit(names: list[str], aligned: dict[str, np.ndarray], target: np.ndarray, fold: np.ndarray) -> dict:
    n_splits = int(fold.max()) + 1
    nested_pred = np.zeros_like(target)
    per_fold = []
    for k in range(n_splits):
        train = fold != k
        test = fold == k
        weights = fit_weight(names, {n: aligned[n][train] for n in names}, target[train])
        train_score = float(ob.tile_rmse(blend_pred(names, {n: aligned[n][train] for n in names}, weights), target[train]).mean())
        test_pred = blend_pred(names, {n: aligned[n][test] for n in names}, weights)
        test_score = float(ob.tile_rmse(test_pred, target[test]).mean())
        nested_pred[test] = test_pred
        per_fold.append({
            "outer_fold": k,
            "n_train_tiles": int(train.sum()),
            "n_test_tiles": int(test.sum()),
            "weights_fit_on_other_folds": weights,
            "train_fit_score": train_score,
            "held_out_score": test_score,
        })

    nested_score = float(ob.tile_rmse(nested_pred, target).mean())

    # naive in-sample score, for the overfitting-gap comparison the docs called for
    in_sample_weights = fit_weight(names, aligned, target)
    in_sample_pred = blend_pred(names, aligned, in_sample_weights)
    in_sample_score = float(ob.tile_rmse(in_sample_pred, target).mean())

    solo_scores = {n: float(ob.tile_rmse(aligned[n], target).mean()) for n in names}
    best_solo_name = min(solo_scores, key=lambda n: solo_scores[n])

    weight_spread = {}
    if names:
        for name in names:
            vals = [f["weights_fit_on_other_folds"].get(name, 0.0) for f in per_fold]
            weight_spread[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "per_fold": vals}

    return {
        "names": names,
        "per_fold": per_fold,
        "nested_score": nested_score,
        "nested_pred": nested_pred,
        "in_sample_weights": in_sample_weights,
        "in_sample_score": in_sample_score,
        "overfitting_gap": in_sample_score - nested_score,
        "solo_scores": solo_scores,
        "best_solo_name": best_solo_name,
        "best_solo_score": solo_scores[best_solo_name],
        "nested_vs_best_solo": nested_score - solo_scores[best_solo_name],
        "in_sample_vs_best_solo": in_sample_score - solo_scores[best_solo_name],
        "weight_fit_spread_across_folds": weight_spread,
    }


def write_oof_sample_metrics(result: dict, target: np.ndarray, satellite: np.ndarray,
                              unique_id: np.ndarray, fold: np.ndarray, loc_map: dict[str, str],
                              out_name: str) -> Path:
    """Emit an oof_sample_metrics.csv-compatible cache so l_eda/exp005's tools can consume the
    nested blend as if it were any other experiment (same schema analyze_oof.py writes)."""
    pred = result["nested_pred"]
    per_tile = np.sqrt(np.square(pred - target).reshape(pred.shape[0], -1).mean(axis=1))
    out_dir = ANALYSIS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = ["fold", "unique_id", "name_location", "satellite_target", "tile_rmse", "target_mean"]
    with (out_dir / "oof_sample_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(pred.shape[0]):
            uid = str(unique_id[i])
            writer.writerow({
                "fold": int(fold[i]),
                "unique_id": uid,
                "name_location": loc_map.get(uid, "unknown"),
                "satellite_target": str(satellite[i]),
                "tile_rmse": float(per_tile[i]),
                "target_mean": float(target[i].mean()),
            })
    return out_dir


def write_oof_pred_npz(result: dict, target: np.ndarray, satellite: np.ndarray,
                        unique_id: np.ndarray, fold: np.ndarray, out_name: str) -> Path:
    """2026-07-30 addition: cache the honest nested-CV blended prediction array in the same
    schema `optimize_blend.py`'s `build_cache`/`g_eda/exp003`'s caches use (pred/target/
    unique_id/satellite/fold, fp16), so downstream post-processing sweeps that expect that
    schema (e.g. `g_eda/exp010/run_causal_smoothing_sweep.py`) can re-tune against the ENSEMBLE's
    own OOF predictions instead of a solo member's -- the whole point being that causal-smoothing
    coefficients fit on `exp038_sigmafixed` solo (as exp065 initially shipped, unavoidably, since
    this cache didn't exist yet) are not guaranteed to be right for a 4-way blend's error
    structure. Written next to `nested_blend_report.json` in OUT_DIR.
    """
    path = OUT_DIR / f"{out_name}_oof_pred.npz"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred=result["nested_pred"].astype(np.float16),
        target=target.astype(np.float16),
        unique_id=unique_id,
        satellite=satellite,
        fold=fold.astype(np.int8),
    )
    return path


def write_report(result: dict, manifest_names: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in result.items() if k != "nested_pred"}
    (OUT_DIR / "nested_blend_report.json").write_text(json.dumps(serializable, indent=2), encoding="utf-8")

    lines = [
        "# Nested (outer-cross-fit) blend report -- g_eda/exp011",
        "",
        f"Sources: {', '.join(manifest_names)}",
        "",
        "## Headline comparison",
        "",
        f"- best solo source: {result['best_solo_name']} = {result['best_solo_score']:.5f}",
        f"- naive in-sample blend fit (what `optimize_blend.py --analyze` reports): "
        f"{result['in_sample_score']:.5f} ({result['in_sample_vs_best_solo']:+.5f} vs best solo)",
        f"- **nested (outer-cross-fit) blend score: {result['nested_score']:.5f} "
        f"({result['nested_vs_best_solo']:+.5f} vs best solo)**",
        f"- overfitting gap (in-sample minus nested): {result['overfitting_gap']:+.5f}",
        "",
        "If the nested score's improvement over best-solo is much smaller than the in-sample "
        "score's (or reverses sign), the in-sample number was fitting fold structure, not signal "
        "-- do not trust it for a submission decision; treat the nested score as the honest one.",
        "",
        "## Per-outer-fold detail",
        "",
        "| fold | n_train | n_test | weights (fit on other folds) | train fit | held-out score |",
        "| ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for f in result["per_fold"]:
        lines.append(f"| {f['outer_fold']} | {f['n_train_tiles']} | {f['n_test_tiles']} | "
                      f"{json.dumps(f['weights_fit_on_other_folds'])} | {f['train_fit_score']:.5f} | "
                      f"{f['held_out_score']:.5f} |")
    lines += ["", "## Weight stability across outer folds", "",
              "Large std here (esp. from the 2-location fold) means the 'optimal' weight is itself "
              "unstable at this sample size -- another way the naive single fit can mislead.", ""]
    for name, spread in result["weight_fit_spread_across_folds"].items():
        lines.append(f"- {name}: mean={spread['mean']:.3f}, std={spread['std']:.3f}, per_fold={spread['per_fold']}")
    lines.append("")
    (OUT_DIR / "NESTED_BLEND.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", default=None,
                         help="override sources.json with an explicit list of manifest source names")
    parser.add_argument("--out-name", default=None,
                         help="analysis dir name for the emitted oof_sample_metrics.csv "
                              "(default: '<first_source>_nested_blend')")
    args = parser.parse_args()

    manifest = ob.load_manifest()
    if args.sources:
        manifest = [ob.source_by_name(name, manifest) for name in args.sources]
    names = [entry["name"] for entry in manifest]

    aligned, target, satellite, unique_id, fold = load_aligned_with_fold(manifest)
    result = nested_cross_fit(names, aligned, target, fold)

    loc_map = unique_id_to_location(names[0])
    out_name = args.out_name or f"{names[0]}_nested_blend"
    out_dir = write_oof_sample_metrics(result, target, satellite, unique_id, fold, loc_map, out_name)
    print(f"wrote {out_dir / 'oof_sample_metrics.csv'} "
          f"(usable as --candidate {out_name} in l_eda/exp005)")

    npz_path = write_oof_pred_npz(result, target, satellite, unique_id, fold, out_name)
    print(f"wrote {npz_path} (usable as a g_eda/exp010-style OOF pred cache for post-process tuning)")

    write_report(result, names)


if __name__ == "__main__":
    main()
