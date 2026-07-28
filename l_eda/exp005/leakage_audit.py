#!/usr/bin/env python3
"""l_eda/exp005 (I-002): gain-concentration audit -- is a candidate's OOF gain geography-shaped?

Motivation: `exp047_sigmafixed` (`doc/public_scores.md`, 2026-07-24) improved OOF by -0.00081 but
regressed public LB by +0.0135, diagnosed as a likely geography shortcut -- `hemisphere` (binary)
crossed with the satellite one-hot lets the model memorize per-(satellite, hemisphere) train-region
climate baselines instead of learning the intended solar-time physics. That diagnosis was made by
hand after the damage was already on the leaderboard. This script makes the check mechanical and
pre-submission: if a candidate's aggregate OOF improvement is actually concentrated in one or two
locations (or one satellite) while most locations are flat or worse, that is the exact signature of
a location-identity shortcut rather than a generalizable feature, whether or not we can name the
mechanism -- eval locations are guaranteed disjoint from train, so anything that looks like
"the model got better at this specific place" cannot be expected to transfer.

Usage:
    python3 l_eda/exp005/leakage_audit.py --baseline exp038_sigmafixed --candidate exp047_sigmafixed

Writes outputs/l_eda/exp005/pairs/{candidate}_vs_{baseline}/leakage_audit.json.
"""

from __future__ import annotations

import argparse
import json

from oof_common import OUT_DIR, align_pair, load_sample_metrics, metric_key

CONCENTRATION_FLAG_THRESHOLD = 0.6  # top-2 locations explaining >60% of total positive gain


def per_location_deltas(loc_tiles: dict[str, list[tuple[float, float]]]) -> list[dict]:
    rows = []
    for loc, pairs in loc_tiles.items():
        a_vals = [a for a, _ in pairs]
        b_vals = [b for _, b in pairs]
        mean_a = sum(a_vals) / len(a_vals)
        mean_b = sum(b_vals) / len(b_vals)
        rows.append({
            "location": loc,
            "n_tiles": len(pairs),
            "baseline_mean": mean_a,
            "candidate_mean": mean_b,
            "improvement": mean_a - mean_b,  # positive = candidate better at this location
        })
    rows.sort(key=lambda r: -r["improvement"])
    return rows


def per_satellite_deltas(rows_a: list[dict], rows_b: list[dict], metric: str) -> list[dict]:
    map_b = {row["unique_id"]: row for row in rows_b}
    by_sat: dict[str, list[tuple[float, float]]] = {}
    for ra in rows_a:
        rb = map_b.get(ra["unique_id"])
        if rb is None:
            continue
        sat = ra["satellite_target"]
        by_sat.setdefault(sat, []).append((float(ra[metric]), float(rb[metric])))
    out = []
    for sat, pairs in sorted(by_sat.items()):
        a_vals = [a for a, _ in pairs]
        b_vals = [b for _, b in pairs]
        out.append({
            "satellite": sat,
            "n_tiles": len(pairs),
            "baseline_mean": sum(a_vals) / len(a_vals),
            "candidate_mean": sum(b_vals) / len(b_vals),
            "improvement": sum(a_vals) / len(a_vals) - sum(b_vals) / len(b_vals),
        })
    return out


def concentration_summary(loc_rows: list[dict]) -> dict:
    positive = [r["improvement"] for r in loc_rows if r["improvement"] > 0]
    total_positive = sum(positive)
    n_better = sum(1 for r in loc_rows if r["improvement"] > 0)
    n_worse = sum(1 for r in loc_rows if r["improvement"] < 0)
    n_flat = len(loc_rows) - n_better - n_worse

    def top_k_share(k: int) -> float:
        if total_positive <= 0:
            return 0.0
        return sum(r["improvement"] for r in loc_rows[:k] if r["improvement"] > 0) / total_positive

    return {
        "n_locations": len(loc_rows),
        "n_locations_better": n_better,
        "n_locations_worse": n_worse,
        "n_locations_flat": n_flat,
        "top1_share_of_positive_gain": top_k_share(1),
        "top2_share_of_positive_gain": top_k_share(2),
        "top3_share_of_positive_gain": top_k_share(3),
        "concentrated": top_k_share(2) > CONCENTRATION_FLAG_THRESHOLD,
    }


def run(baseline: str, candidate: str) -> dict:
    rows_a = load_sample_metrics(baseline)
    rows_b = load_sample_metrics(candidate)
    metric = metric_key(rows_a)
    loc_tiles, diagnostics = align_pair(rows_a, rows_b, metric)

    loc_rows = per_location_deltas(loc_tiles)
    sat_rows = per_satellite_deltas(rows_a, rows_b, metric)
    concentration = concentration_summary(loc_rows)

    return {
        "baseline": baseline,
        "candidate": candidate,
        "metric": metric,
        "diagnostics": diagnostics,
        "per_location": loc_rows,
        "per_satellite": sat_rows,
        "concentration": concentration,
    }


def write_report(result: dict) -> None:
    pair_dir = OUT_DIR / "pairs" / f"{result['candidate']}_vs_{result['baseline']}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "leakage_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    c = result["concentration"]
    lines = [
        f"# Gain-concentration audit: {result['candidate']} vs {result['baseline']}",
        "",
        f"{c['n_locations']} locations: {c['n_locations_better']} better, {c['n_locations_worse']} worse, "
        f"{c['n_locations_flat']} flat.",
        "",
        f"- top-1 location share of total positive gain: {c['top1_share_of_positive_gain']:.1%}",
        f"- top-2 location share of total positive gain: {c['top2_share_of_positive_gain']:.1%}",
        f"- top-3 location share of total positive gain: {c['top3_share_of_positive_gain']:.1%}",
        "",
        f"**{'CONCENTRATED -- gain is not spread across locations; treat as a possible location-specific shortcut, not generalizable physics.' if c['concentrated'] else 'Diffuse -- gain is spread across multiple locations, consistent with a generalizable effect.'}**",
        "",
        "## Per-location breakdown (sorted by improvement, positive = candidate better)",
        "",
        "| location | n_tiles | baseline | candidate | improvement |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for r in result["per_location"]:
        lines.append(f"| {r['location']} | {r['n_tiles']} | {r['baseline_mean']:.4f} | "
                      f"{r['candidate_mean']:.4f} | {r['improvement']:+.4f} |")
    lines += ["", "## Per-satellite breakdown", "", "| satellite | n_tiles | baseline | candidate | improvement |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for r in result["per_satellite"]:
        lines.append(f"| {r['satellite']} | {r['n_tiles']} | {r['baseline_mean']:.4f} | "
                      f"{r['candidate_mean']:.4f} | {r['improvement']:+.4f} |")
    lines.append("")
    (pair_dir / "LEAKAGE_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    result = run(args.baseline, args.candidate)
    write_report(result)


if __name__ == "__main__":
    main()
