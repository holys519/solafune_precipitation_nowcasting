#!/usr/bin/env python3
"""l_eda/exp005 (I-002): location-cluster bootstrap CI for a candidate-vs-baseline OOF delta.

Motivation (see `doc/public_scores.md` Observations, 2026-07-24 session): three recent
candidates (exp050_sigmafixed, exp047_sigmafixed, exp055's blend) all showed an OOF improvement
that failed to transfer to the public LB -- one of them (exp047_sigmafixed) inverted into the
single worst regression of the project. The common structural cause is that this project's 5-fold
OOF is a *point estimate* built from only 20 train locations (as few as 2 locations in fold 0),
so a delta of a few thousandths is frequently indistinguishable from which locations happened to
land in which fold, not from real model quality.

This script quantifies that: it treats each *location* (not each tile) as the natural
resampling unit (a cluster/block bootstrap, since tiles within one location are not
independent draws -- they share the same fold, satellite, climate regime), resamples the 20
locations with replacement many times, and reports the empirical distribution of the candidate's
delta over baseline. A delta whose bootstrap CI straddles zero should not move the champion,
regardless of how good the single point estimate looks.

Usage:
    python3 l_eda/exp005/bootstrap_ci.py --baseline exp038_sigmafixed --candidate exp050_sigmafixed
    python3 l_eda/exp005/bootstrap_ci.py --baseline exp038_sigmafixed --candidate exp047_sigmafixed --n-boot 20000

Reads outputs/analysis/{exp}/oof_sample_metrics.csv for both experiments (already produced by
their own analyze_oof.py -- no training or inference here). Writes
outputs/l_eda/exp005/pairs/{candidate}_vs_{baseline}/bootstrap_ci.json.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from oof_common import (
    NOISE_FLOOR,
    OUT_DIR,
    align_pair,
    load_sample_metrics,
    location_cluster_sums,
    metric_key,
)


def bootstrap_deltas(sum_a: np.ndarray, sum_b: np.ndarray, n: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Cluster (location-level) bootstrap of the tile-weighted mean(candidate) - mean(baseline).

    Each draw resamples len(locations) locations with replacement and pools *all* of that
    location's original tiles -- this preserves the official metric's tile-equal-weighting while
    perturbing exactly the thing E-4 identified as the dominant noise source: which locations you
    got. Vectorized: O(n_boot * n_locations) instead of O(n_boot * n_tiles).
    """
    n_loc = len(n)
    idx = rng.integers(0, n_loc, size=(n_boot, n_loc))
    total_a = sum_a[idx].sum(axis=1)
    total_b = sum_b[idx].sum(axis=1)
    total_n = n[idx].sum(axis=1)
    return (total_b - total_a) / total_n


def run(baseline: str, candidate: str, n_boot: int, seed: int) -> dict:
    rows_a = load_sample_metrics(baseline)
    rows_b = load_sample_metrics(candidate)
    metric = metric_key(rows_a)
    loc_tiles, diagnostics = align_pair(rows_a, rows_b, metric)
    locations, sum_a, sum_b, n = location_cluster_sums(loc_tiles)

    point_delta = float((sum_b.sum() - sum_a.sum()) / n.sum())
    baseline_mean = float(sum_a.sum() / n.sum())
    candidate_mean = float(sum_b.sum() / n.sum())

    rng = np.random.default_rng(seed)
    deltas = bootstrap_deltas(sum_a, sum_b, n, n_boot, rng)

    ci = {
        "50": [float(np.percentile(deltas, 25)), float(np.percentile(deltas, 75))],
        "80": [float(np.percentile(deltas, 10)), float(np.percentile(deltas, 90))],
        "90": [float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))],
    }
    prob_candidate_better = float((deltas < 0).mean())

    result = {
        "metric": metric,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_mean": baseline_mean,
        "candidate_mean": candidate_mean,
        "point_delta": point_delta,
        "n_boot": n_boot,
        "seed": seed,
        "n_locations": len(locations),
        "locations": locations,
        "ci": ci,
        "prob_candidate_better": prob_candidate_better,
        "noise_floor": NOISE_FLOOR,
        "below_noise_floor": abs(point_delta) < NOISE_FLOOR,
        "ci80_excludes_zero": ci["80"][1] < 0 or ci["80"][0] > 0,
        "diagnostics": diagnostics,
    }
    return result


def write_report(result: dict) -> None:
    pair_dir = OUT_DIR / "pairs" / f"{result['candidate']}_vs_{result['baseline']}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "bootstrap_ci.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    d = result
    lines = [
        f"# Bootstrap CI: {d['candidate']} vs {d['baseline']}",
        "",
        f"Metric: `{d['metric']}` (lower is better). N locations: {d['n_locations']} "
        f"(this is the real sample size for generalization claims -- not the {d['diagnostics']['n_shared_tiles']} tiles).",
        "",
        f"- baseline mean: {d['baseline_mean']:.5f}",
        f"- candidate mean: {d['candidate_mean']:.5f}",
        f"- point delta (candidate - baseline): {d['point_delta']:+.5f} "
        f"({'below' if d['below_noise_floor'] else 'above'} the {d['noise_floor']} noise floor from l_eda/exp003)",
        "",
        "## Location-cluster bootstrap (resamples which 20 locations you'd have gotten)",
        "",
        f"- 50% CI: [{d['ci']['50'][0]:+.5f}, {d['ci']['50'][1]:+.5f}]",
        f"- 80% CI: [{d['ci']['80'][0]:+.5f}, {d['ci']['80'][1]:+.5f}]",
        f"- 90% CI: [{d['ci']['90'][0]:+.5f}, {d['ci']['90'][1]:+.5f}]",
        f"- P(candidate better) over {d['n_boot']} resamples: {d['prob_candidate_better']:.3f}",
        "",
        f"**Verdict: {'80% CI excludes zero' if d['ci80_excludes_zero'] else '80% CI INCLUDES ZERO -- delta is not distinguishable from location-composition noise at this sample size'}.**",
        "",
        "## Alignment diagnostics",
        "",
        f"- shared tiles: {d['diagnostics']['n_shared_tiles']}",
        f"- dropped (baseline-only): {d['diagnostics']['n_dropped_a_only']}, "
        f"dropped (candidate-only): {d['diagnostics']['n_dropped_b_only']}",
        f"- location mismatches (same unique_id, different name_location between runs): "
        f"{d['diagnostics']['n_location_mismatches']}"
        + (" -- INVESTIGATE: these two runs may not share a fold/location split" if d['diagnostics']['n_location_mismatches'] else ""),
        "",
    ]
    (pair_dir / "BOOTSTRAP_CI.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="experiment name under outputs/analysis/ (the current champion)")
    parser.add_argument("--candidate", required=True, help="experiment name under outputs/analysis/ (the challenger)")
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    result = run(args.baseline, args.candidate, args.n_boot, args.seed)
    write_report(result)


if __name__ == "__main__":
    main()
