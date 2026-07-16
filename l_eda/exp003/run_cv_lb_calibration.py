#!/usr/bin/env python3
"""E-3 (Round 5 plan): calibrate CV metrics against the public leaderboard.

Uses only the standard library so it runs on the login node without a container.
For every experiment with both a public RMSE and OOF sample metrics, computes four
candidate CV predictors and measures how well each predicts public-LB ordering:

- oof_tile_rmse:        mean per-tile RMSE over all OOF samples (the current default metric)
- oof_sat_weighted:     per-satellite tile RMSE reweighted to the test mix
                        (himawari .39 / meteosat .39 / goes .22, discussion Finding 10)
- fold0_tile_rmse:      fold 0 only (the exp028-032 screening shortcut)
- fold04_tile_rmse:     mean of folds 0 and 4 (the Round 5 proposed screening pair)

Outputs (outputs/l_eda/exp003/):
- cv_lb_pairs.csv       one row per experiment with all predictors + public RMSE
- CV_LB_CALIBRATION.md  rank correlations, linear fits, residuals, and the implied
                        LB-noise threshold for accept/reject decisions
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_DIR / "outputs" / "analysis"
OUT_DIR = PROJECT_DIR / "outputs" / "l_eda" / "exp003"

SATELLITE_WEIGHTS = {"himawari": 0.39, "meteosat": 0.39, "goes": 0.22}

# Public scores from doc/public_scores.md (2026-07-10) plus exp026's anchor recorded in
# outputs/analysis/exp033/analysis_summary.json. Model-level submissions only: exp014/exp026
# are overlap-patch post-processing and exp007 lacks per-sample OOF metrics, so they cannot
# calibrate a model-OOF predictor. exp015 is isotonic post-processing on exp009 checkpoints;
# it is kept but flagged because its OOF CSV reflects the uncalibrated model.
PUBLIC_SCORES = {
    "exp003": (0.7522576632294679, "model"),
    "exp004": (0.7252533726905589, "model"),
    "exp005": (0.7445524878914139, "model"),
    "exp006": (0.7450324204392412, "model"),
    "exp008": (0.7250185237499447, "postprocess-on-exp004"),
    "exp009": (0.7153438899106017, "model"),
    "exp010": (0.7348731115909746, "model"),
    "exp011": (0.7232307883574975, "model"),
    "exp015": (0.7096658388930687, "postprocess-on-exp009"),
    "exp016": (0.6977629323809645, "model"),
    "exp017": (0.6997414980565597, "model"),
    "exp018": (0.6929495140301676, "model"),
}


def tile_rmse_column(header: list[str]) -> str:
    return "tile_rmse" if "tile_rmse" in header else "rmse"


def load_sample_metrics(exp: str) -> list[dict[str, str]] | None:
    path = ANALYSIS_DIR / exp / "oof_sample_metrics.csv"
    if not path.exists():
        return None
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def predictors_for(rows: list[dict[str, str]]) -> dict[str, float]:
    col = tile_rmse_column(list(rows[0].keys()))
    by_sat: dict[str, list[float]] = {}
    by_fold: dict[str, list[float]] = {}
    all_values: list[float] = []
    for row in rows:
        value = float(row[col])
        all_values.append(value)
        by_sat.setdefault(row["satellite_target"], []).append(value)
        by_fold.setdefault(row["fold"], []).append(value)
    sat_weighted = sum(SATELLITE_WEIGHTS[sat] * mean(vals) for sat, vals in by_sat.items())
    return {
        "oof_tile_rmse": mean(all_values),
        "oof_sat_weighted": sat_weighted,
        "fold0_tile_rmse": mean(by_fold["0"]),
        "fold04_tile_rmse": (mean(by_fold["0"]) + mean(by_fold["4"])) / 2.0,
    }


def ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    for rank, idx in enumerate(order):
        result[idx] = float(rank)
    return result


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    vy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return cov / (vx * vy) if vx and vy else float("nan")


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(ranks(xs), ranks(ys))


def kendall(xs: list[float], ys: list[float]) -> tuple[float, int, int]:
    concordant = discordant = 0
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            sign = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if sign > 0:
                concordant += 1
            elif sign < 0:
                discordant += 1
    total = concordant + discordant
    tau = (concordant - discordant) / total if total else float("nan")
    return tau, concordant, discordant


def linear_fit(xs: list[float], ys: list[float]) -> tuple[float, float, list[float]]:
    mx, my = mean(xs), mean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var else 0.0
    intercept = my - slope * mx
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    return slope, intercept, residuals


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table: list[dict[str, object]] = []
    for exp, (public, kind) in sorted(PUBLIC_SCORES.items()):
        rows = load_sample_metrics(exp)
        if rows is None:
            print(f"skip {exp}: no oof_sample_metrics.csv")
            continue
        entry: dict[str, object] = {"experiment": exp, "kind": kind, "public_rmse": public}
        entry.update(predictors_for(rows))
        table.append(entry)

    fields = ["experiment", "kind", "public_rmse", "oof_tile_rmse", "oof_sat_weighted",
              "fold0_tile_rmse", "fold04_tile_rmse"]
    with (OUT_DIR / "cv_lb_pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    predictor_names = ["oof_tile_rmse", "oof_sat_weighted", "fold0_tile_rmse", "fold04_tile_rmse"]
    public = [float(row["public_rmse"]) for row in table]
    model_rows = [row for row in table if row["kind"] == "model"]
    model_public = [float(row["public_rmse"]) for row in model_rows]

    stats: dict[str, dict[str, object]] = {}
    lines = ["# E-3: CV -> Public LB calibration", "",
             f"Pairs used: {len(table)} (of which pure model submissions: {len(model_rows)})", "",
             "## Pairs", "",
             "| Exp | kind | public | oof | sat-weighted | fold0 | fold0+4 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in table:
        lines.append(
            f"| {row['experiment']} | {row['kind']} | {row['public_rmse']:.4f} | "
            f"{row['oof_tile_rmse']:.4f} | {row['oof_sat_weighted']:.4f} | "
            f"{row['fold0_tile_rmse']:.4f} | {row['fold04_tile_rmse']:.4f} |"
        )
    lines += ["", "## Predictor quality (all pairs)", "",
              "| Predictor | Spearman | Kendall tau | concordant/discordant | fit residual std | max \\|resid\\| |",
              "| --- | ---: | ---: | --- | ---: | ---: |"]
    for name in predictor_names:
        xs = [float(row[name]) for row in table]
        tau, con, dis = kendall(xs, public)
        slope, intercept, residuals = linear_fit(xs, public)
        resid_std = math.sqrt(mean([r * r for r in residuals]))
        stats[name] = {
            "spearman": spearman(xs, public), "kendall": tau,
            "slope": slope, "intercept": intercept, "resid_std": resid_std,
        }
        lines.append(
            f"| {name} | {stats[name]['spearman']:.3f} | {tau:.3f} | {con}/{dis} | "
            f"{resid_std:.4f} | {max(abs(r) for r in residuals):.4f} |"
        )
    lines += ["", "## Predictor quality (pure model pairs only)", "",
              "| Predictor | Spearman | Kendall tau |", "| --- | ---: | ---: |"]
    for name in predictor_names:
        xs = [float(row[name]) for row in model_rows]
        tau, _, _ = kendall(xs, model_public)
        lines.append(f"| {name} | {spearman(xs, model_public):.3f} | {tau:.3f} |")

    best = min(stats, key=lambda k: 1 - abs(stats[k]["spearman"]))
    lines += ["", "## Reading", "",
              f"- Highest-|Spearman| predictor: **{best}**.",
              "- `fit residual std` is the empirical LB-noise scale: OOF-predicted LB deltas smaller",
              "  than ~1 residual std cannot be trusted from CV alone.",
              "- Discordant pairs above identify exactly which historical A/Bs the predictor",
              "  would have called wrong — inspect them before changing the accept metric.", ""]
    (OUT_DIR / "CV_LB_CALIBRATION.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT_DIR / "calibration_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {OUT_DIR}/cv_lb_pairs.csv, CV_LB_CALIBRATION.md, calibration_stats.json")


if __name__ == "__main__":
    main()
