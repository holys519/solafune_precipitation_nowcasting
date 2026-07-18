#!/usr/bin/env python3
"""E-4 (Round 5 plan): anatomy of the bad folds (fold1 0.75 / fold3 0.79 vs fold0 0.29).

Question: is the fold variance regime-driven (heavy-rain monsoon locations are intrinsically
harder) or location-idiosyncratic? Stratifies exp018's OOF per-tile metrics by location and
by target-mean regime bins, and computes what each fold's tile_rmse would be if every fold
had fold0's regime mix (fixed-mix counterfactual). Stdlib only.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "outputs" / "l_eda" / "exp004"

REGIME_EDGES = [0.0, 0.01, 0.1, 0.3, 1.0, 100.0]  # target_mean bins (mm/hr tile mean)


def regime_of(target_mean: float) -> int:
    for i in range(len(REGIME_EDGES) - 1):
        if REGIME_EDGES[i] <= target_mean < REGIME_EDGES[i + 1]:
            return i
    return len(REGIME_EDGES) - 2


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="exp018", help="experiment name under outputs/analysis/")
    args = parser.parse_args()
    exp = args.exp
    sample_csv = PROJECT_DIR / "outputs" / "analysis" / exp / "oof_sample_metrics.csv"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(sample_csv.open(newline="")))
    col = "tile_rmse" if "tile_rmse" in rows[0] else "rmse"

    by_fold: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_fold[row["fold"]].append(row)

    # per-fold regime mix and within-regime error
    regime_share: dict[str, list[float]] = {}
    regime_err: dict[str, list[float]] = {}
    n_regimes = len(REGIME_EDGES) - 1
    for fold, fold_rows in sorted(by_fold.items()):
        counts = [0] * n_regimes
        errs: list[list[float]] = [[] for _ in range(n_regimes)]
        for row in fold_rows:
            r = regime_of(float(row["target_mean"]))
            counts[r] += 1
            errs[r].append(float(row[col]))
        total = sum(counts)
        regime_share[fold] = [c / total for c in counts]
        regime_err[fold] = [mean(e) for e in errs]

    lines = [f"# E-4: fold variance anatomy ({exp} OOF)", "",
             "Regime bins by tile target_mean (mm/hr): "
             f"{REGIME_EDGES[:-1]} (last bin open)", "",
             "## Per-fold regime mix (share) and within-regime tile_rmse", "",
             "| fold | actual | " + " | ".join(f"share r{i}" for i in range(n_regimes)) +
             " | " + " | ".join(f"rmse r{i}" for i in range(n_regimes)) + " | fixed-mix* |",
             "|" + " ---: |" * (2 + 2 * n_regimes + 1)]
    mix0 = regime_share["0"]
    for fold in sorted(by_fold):
        actual = mean([float(r[col]) for r in by_fold[fold]])
        fixed = sum(share * err for share, err in zip(mix0, regime_err[fold])
                    if err == err)  # NaN-safe
        shares = " | ".join(f"{s:.2f}" for s in regime_share[fold])
        errs = " | ".join(f"{e:.3f}" if e == e else "-" for e in regime_err[fold])
        lines.append(f"| {fold} | {actual:.4f} | {shares} | {errs} | {fixed:.4f} |")
    lines += ["", "*fixed-mix = this fold's within-regime errors weighted by FOLD0's regime mix.",
              "If fixed-mix collapses the fold spread, the variance is regime-composition;",
              "if the spread persists, the fold's locations are intrinsically harder per regime.", ""]

    # per-location detail
    by_loc: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_loc[row["name_location"]].append(row)
    loc_rows = []
    for location, loc_list in sorted(by_loc.items()):
        wet = [r for r in loc_list if float(r["target_mean"]) > 0.1]
        loc_rows.append({
            "fold": loc_list[0]["fold"], "location": location,
            "satellite": loc_list[0]["satellite_target"], "samples": len(loc_list),
            "tile_rmse": round(mean([float(r[col]) for r in loc_list]), 4),
            "target_mean": round(mean([float(r["target_mean"]) for r in loc_list]), 4),
            "share_target_mean_gt0.1": round(len(wet) / len(loc_list), 3),
            "wet_tile_rmse": round(mean([float(r[col]) for r in wet]), 4) if wet else "",
        })
    loc_rows.sort(key=lambda r: -float(r["tile_rmse"]))
    with (OUT_DIR / f"location_anatomy_{exp}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(loc_rows[0].keys()))
        writer.writeheader()
        writer.writerows(loc_rows)
    lines += ["## Locations ranked by tile_rmse (top 10)", "",
              "| fold | location | sat | tile_rmse | target_mean | share>0.1 |",
              "| --- | --- | --- | ---: | ---: | ---: |"]
    for row in loc_rows[:10]:
        lines.append(f"| {row['fold']} | {row['location']} | {row['satellite']} | "
                     f"{row['tile_rmse']} | {row['target_mean']} | {row['share_target_mean_gt0.1']} |")

    (OUT_DIR / f"FOLD_ANATOMY_{exp}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
