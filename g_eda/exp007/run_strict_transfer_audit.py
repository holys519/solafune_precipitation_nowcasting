#!/usr/bin/env python3
"""Audit strict/green CV-to-Public transfer around exp038.

This analysis is deliberately sample-summary only.  It never reads evaluation
targets or prediction rasters and cannot create a submission artifact.

Outputs
-------
* paired_deltas.csv: exp038 minus each comparator by OOF subgroup.
* location_bootstrap.csv: location-cluster bootstrap intervals for paired deltas.
* metadata_reweighted_scores.csv: OOF scores reweighted to evaluation metadata.
* evaluation_prediction_summary.csv: aggregate, target-free prediction statistics.
* public_cv_comparison.csv: post-hoc CV/Public comparison for interpretation only.
* TRANSFER_AUDIT.md and summary.json: concise findings and machine-readable summary.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = PROJECT_DIR / "outputs" / "analysis"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp007"
TRAIN_CSV = PROJECT_DIR / "data" / "train_dataset" / "train_dataset.csv"
EVAL_CSV = PROJECT_DIR / "data" / "evaluation_dataset" / "evaluation_target.csv"

EXPERIMENTS = (
    "exp011",
    "exp016",
    "exp017",
    "exp018",
    "exp035_no_dilation",
    "exp038",
)
REFERENCE = "exp038"
PUBLIC_SCORES = {
    "exp011": 0.7232307883574975,
    "exp016": 0.6977629323809645,
    "exp017": 0.6997414980565597,
    "exp018": 0.6929495140301676,
    "exp035_no_dilation": 0.6860146267326392,
    "exp038": 0.6891638997287517,
}
PUBLIC_SOURCES = {
    "exp011": "doc/public_scores.md",
    "exp016": "doc/public_scores.md",
    "exp017": "doc/public_scores.md",
    "exp018": "doc/public_scores.md",
    "exp035_no_dilation": "outputs/l_eda/exp003/cv_lb_pairs.csv",
    "exp038": "user report 2026-07-18 06:01:30 UTC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def observation_count(value: object) -> int:
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"expected observation list, got {type(parsed)!r}")
    return len(parsed)


def metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="raise")
    frame["hour_bin"] = (frame["datetime"].dt.hour // 6).astype(int)
    frame["observation_count"] = frame[
        "last_30_minutes_observation_filename"
    ].map(observation_count)
    return frame[
        [
            "unique_id",
            "name_location",
            "satellite_target",
            "datetime",
            "hour_bin",
            "observation_count",
        ]
    ].copy()


def target_amount_bin(values: pd.Series) -> pd.Categorical:
    categories = [
        "dry",
        "light_(0,0.1)",
        "moderate_[0.1,0.5)",
        "heavy_[0.5,1)",
        "extreme_[1,inf)",
    ]
    data = values.to_numpy(dtype=np.float64)
    labels = np.full(len(data), categories[-1], dtype=object)
    labels[data == 0.0] = categories[0]
    labels[(data > 0.0) & (data < 0.1)] = categories[1]
    labels[(data >= 0.1) & (data < 0.5)] = categories[2]
    labels[(data >= 0.5) & (data < 1.0)] = categories[3]
    return pd.Categorical(
        labels,
        categories=categories,
        ordered=True,
    )


def load_oof() -> tuple[pd.DataFrame, dict[str, Path]]:
    merged: pd.DataFrame | None = None
    paths: dict[str, Path] = {}
    common = [
        "unique_id",
        "fold",
        "name_location",
        "satellite_target",
        "datetime",
        "target_mean",
    ]
    for experiment in EXPERIMENTS:
        path = ANALYSIS_DIR / experiment / "oof_sample_metrics.csv"
        paths[experiment] = path
        frame = pd.read_csv(path)
        required = set(common + ["tile_rmse", "pred_mean"])
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        if frame["unique_id"].duplicated().any():
            raise ValueError(f"{path}: duplicate unique_id")
        selected = frame[common + ["tile_rmse", "pred_mean"]].copy()
        selected = selected.rename(
            columns={
                "tile_rmse": f"tile_rmse__{experiment}",
                "pred_mean": f"pred_mean__{experiment}",
            }
        )
        if merged is None:
            merged = selected
        else:
            merged = merged.merge(
                selected,
                on=common,
                how="inner",
                validate="one_to_one",
            )
    assert merged is not None
    expected = len(pd.read_csv(paths[REFERENCE], usecols=["unique_id"]))
    if len(merged) != expected:
        raise ValueError(f"OOF alignment retained {len(merged)} of {expected} rows")
    merged["datetime"] = pd.to_datetime(merged["datetime"], errors="raise")
    merged["target_amount_bin"] = target_amount_bin(merged["target_mean"])
    return merged, paths


def paired_group_rows(oof: pd.DataFrame) -> list[dict[str, object]]:
    groupings: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("fold", ["fold"]),
        ("satellite", ["satellite_target"]),
        ("location", ["name_location"]),
        ("target_amount", ["target_amount_bin"]),
        ("satellite_x_amount", ["satellite_target", "target_amount_bin"]),
    ]
    rows: list[dict[str, object]] = []
    reference_values = oof[f"tile_rmse__{REFERENCE}"]
    for comparator in EXPERIMENTS:
        if comparator == REFERENCE:
            continue
        delta = reference_values - oof[f"tile_rmse__{comparator}"]
        work = oof.copy()
        work["paired_delta"] = delta
        for group_type, columns in groupings:
            groups = [("all", work)] if not columns else work.groupby(columns, observed=True, sort=True)
            for key, group in groups:
                if isinstance(key, tuple):
                    label = " | ".join(map(str, key))
                else:
                    label = str(key)
                rows.append(
                    {
                        "reference": REFERENCE,
                        "comparator": comparator,
                        "group_type": group_type,
                        "group": label,
                        "n": len(group),
                        "reference_score": group[f"tile_rmse__{REFERENCE}"].mean(),
                        "comparator_score": group[f"tile_rmse__{comparator}"].mean(),
                        "paired_delta_reference_minus_comparator": group["paired_delta"].mean(),
                        "reference_win_rate": (group["paired_delta"] < 0).mean(),
                    }
                )
    return rows


def stratified_location_bootstrap(
    oof: pd.DataFrame, comparator: str, n_bootstrap: int, rng: np.random.Generator
) -> dict[str, object]:
    work = oof[
        ["satellite_target", "name_location", f"tile_rmse__{REFERENCE}", f"tile_rmse__{comparator}"]
    ].copy()
    work["delta"] = work[f"tile_rmse__{REFERENCE}"] - work[f"tile_rmse__{comparator}"]
    clusters = (
        work.groupby(["satellite_target", "name_location"], sort=True)["delta"]
        .agg(["sum", "count"])
        .reset_index()
    )
    strata = []
    for satellite, group in clusters.groupby("satellite_target", sort=True):
        strata.append((satellite, group["sum"].to_numpy(), group["count"].to_numpy()))
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        total_sum = 0.0
        total_count = 0
        for _, sums, counts in strata:
            draw = rng.integers(0, len(sums), size=len(sums))
            total_sum += float(sums[draw].sum())
            total_count += int(counts[draw].sum())
        samples[i] = total_sum / total_count
    point = float(work["delta"].mean())
    return {
        "reference": REFERENCE,
        "comparator": comparator,
        "n_samples": len(work),
        "n_locations": len(clusters),
        "n_bootstrap": n_bootstrap,
        "paired_delta_reference_minus_comparator": point,
        "ci_2p5": float(np.quantile(samples, 0.025)),
        "ci_50": float(np.quantile(samples, 0.5)),
        "ci_97p5": float(np.quantile(samples, 0.975)),
        "bootstrap_probability_reference_better": float((samples < 0).mean()),
    }


def group_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].astype(str).agg("|".join, axis=1)


def metadata_weights(
    train: pd.DataFrame, evaluation: pd.DataFrame, columns: list[str]
) -> tuple[np.ndarray, float, float, int]:
    train_key = group_key(train, columns)
    eval_key = group_key(evaluation, columns)
    train_share = train_key.value_counts(normalize=True)
    eval_share = eval_key.value_counts(normalize=True)
    missing = sorted(set(eval_share.index) - set(train_share.index))
    unsupported_share = float(eval_share.reindex(missing).sum()) if missing else 0.0
    ratio = (eval_share / train_share).replace([np.inf, -np.inf], np.nan).dropna()
    weights = train_key.map(ratio).fillna(0.0).to_numpy(dtype=np.float64)
    if weights.sum() <= 0:
        raise ValueError(f"zero metadata weights for {columns}")
    ess = float(weights.sum() ** 2 / np.square(weights).sum())
    covered_eval_share = 1.0 - unsupported_share
    return weights, ess, covered_eval_share, len(missing)


def metadata_reweight_rows(oof: pd.DataFrame, train: pd.DataFrame, evaluation: pd.DataFrame) -> list[dict[str, object]]:
    meta = train.merge(
        oof[["unique_id"]], on="unique_id", how="inner", validate="one_to_one"
    )
    if len(meta) != len(oof):
        raise ValueError(f"train metadata alignment retained {len(meta)} of {len(oof)} rows")
    order = oof[["unique_id"]].merge(meta, on="unique_id", how="left", validate="one_to_one")
    schemes = {
        "unweighted": [],
        "satellite": ["satellite_target"],
        "satellite_x_6h": ["satellite_target", "hour_bin"],
        "satellite_x_observation_count": ["satellite_target", "observation_count"],
        "satellite_x_6h_x_observation_count": [
            "satellite_target",
            "hour_bin",
            "observation_count",
        ],
    }
    rows: list[dict[str, object]] = []
    for scheme, columns in schemes.items():
        if columns:
            weights, ess, covered_share, missing_groups = metadata_weights(order, evaluation, columns)
        else:
            weights = np.ones(len(order), dtype=np.float64)
            ess = float(len(order))
            covered_share = 1.0
            missing_groups = 0
        for experiment in EXPERIMENTS:
            values = oof[f"tile_rmse__{experiment}"].to_numpy(dtype=np.float64)
            score = float(np.average(values, weights=weights))
            rows.append(
                {
                    "scheme": scheme,
                    "experiment": experiment,
                    "score": score,
                    "delta_vs_exp038": score - float(
                        np.average(oof[f"tile_rmse__{REFERENCE}"], weights=weights)
                    ),
                    "effective_sample_size": ess,
                    "covered_evaluation_share": covered_share,
                    "unsupported_evaluation_groups": missing_groups,
                }
            )
    return rows


def evaluation_prediction_rows() -> tuple[list[dict[str, object]], dict[str, Path]]:
    rows: list[dict[str, object]] = []
    paths: dict[str, Path] = {}
    reference: pd.DataFrame | None = None
    for experiment in EXPERIMENTS:
        path = ANALYSIS_DIR / experiment / "evaluation_prediction_summary.csv"
        paths[experiment] = path
        frame = pd.read_csv(path)
        if frame["unique_id"].duplicated().any():
            raise ValueError(f"{path}: duplicate unique_id")
        if reference is None:
            reference = frame[["unique_id", "satellite_target", "name_location"]].copy()
        else:
            aligned = reference.merge(
                frame[["unique_id", "satellite_target", "name_location"]],
                on=["unique_id", "satellite_target", "name_location"],
                how="inner",
                validate="one_to_one",
            )
            if len(aligned) != len(reference):
                raise ValueError(f"evaluation summaries do not align for {experiment}")
        groupings = [("overall", "all", frame)] + [
            ("satellite", satellite, group)
            for satellite, group in frame.groupby("satellite_target", sort=True)
        ]
        for group_type, group, subset in groupings:
            rows.append(
                {
                    "experiment": experiment,
                    "group_type": group_type,
                    "group": group,
                    "n": len(subset),
                    "pred_mean_mean": subset["pred_mean"].mean(),
                    "pred_mean_median": subset["pred_mean"].median(),
                    "pred_std_mean": subset["pred_std"].mean(),
                    "pred_max_mean": subset["pred_max"].mean(),
                    "pred_positive_ratio_mean": subset["pred_positive_ratio"].mean(),
                }
            )
    return rows, paths


def public_cv_rows(oof: pd.DataFrame) -> list[dict[str, object]]:
    reference_cv = float(oof[f"tile_rmse__{REFERENCE}"].mean())
    reference_public = PUBLIC_SCORES[REFERENCE]
    rows = []
    for experiment in EXPERIMENTS:
        cv = float(oof[f"tile_rmse__{experiment}"].mean())
        public = PUBLIC_SCORES[experiment]
        rows.append(
            {
                "experiment": experiment,
                "oof_tile_rmse": cv,
                "public_rmse": public,
                "public_minus_oof": public - cv,
                "oof_delta_vs_exp038": cv - reference_cv,
                "public_delta_vs_exp038": public - reference_public,
                "public_source": PUBLIC_SOURCES[experiment],
                "post_hoc_only": True,
            }
        )
    return rows


def f6(value: float) -> str:
    return f"{value:.6f}"


def write_report(
    path: Path,
    paired: pd.DataFrame,
    bootstrap: pd.DataFrame,
    reweighted: pd.DataFrame,
    eval_predictions: pd.DataFrame,
    public_cv: pd.DataFrame,
) -> None:
    overall = paired[paired["group_type"] == "overall"].set_index("comparator")
    satellite = paired[paired["group_type"] == "satellite"]
    boot = bootstrap.set_index("comparator")
    eval_overall = eval_predictions[eval_predictions["group_type"] == "overall"].set_index("experiment")
    reweight_pivot = reweighted.pivot(index="scheme", columns="experiment", values="score")
    lines = [
        "# exp007 strict CV→Public transfer audit",
        "",
        "## Scope",
        "",
        "This is a target-free transfer audit except for existing train OOF labels. Evaluation",
        "targets and prediction rasters are never read. Public scores are shown only in the final",
        "post-hoc table and are not used to fit weights or select a model.",
        "",
        "## Main findings",
        "",
        f"- exp038 OOF is `{f6(float(oof_score := public_cv.set_index('experiment').loc[REFERENCE, 'oof_tile_rmse']))}` and Public is `{f6(PUBLIC_SCORES[REFERENCE])}`.",
        f"- Against exp011, exp038 improves OOF by `{f6(float(overall.loc['exp011', 'paired_delta_reference_minus_comparator']))}` (exp038−exp011; negative is better), and every satellite subgroup improves.",
        f"- Against exp018, exp038 changes OOF by `{f6(float(overall.loc['exp018', 'paired_delta_reference_minus_comparator']))}`: GOES improves but Himawari and Meteosat worsen.",
        f"- Against exp035_no_dilation, exp038 changes OOF by `{f6(float(overall.loc['exp035_no_dilation', 'paired_delta_reference_minus_comparator']))}` and loses in all three satellite subgroups.",
        f"- Location-cluster bootstrap P(exp038 better) is `{float(boot.loc['exp011', 'bootstrap_probability_reference_better']):.3f}` vs exp011, `{float(boot.loc['exp018', 'bootstrap_probability_reference_better']):.3f}` vs exp018, and `{float(boot.loc['exp035_no_dilation', 'bootstrap_probability_reference_better']):.3f}` vs exp035_no_dilation.",
        "- Reweighting OOF to evaluation satellite / time-of-day / observation-count metadata does not reproduce the exp038 Public inversion against exp018 or exp035_no_dilation.",
        f"- exp038 has a lower target-free evaluation mean prediction (`{f6(float(eval_overall.loc[REFERENCE, 'pred_mean_mean']))}`) than exp018 (`{f6(float(eval_overall.loc['exp018', 'pred_mean_mean']))}`) and exp035_no_dilation (`{f6(float(eval_overall.loc['exp035_no_dilation', 'pred_mean_mean']))}`). This is a hypothesis-generating observation, not evidence that lower amplitude caused the Public result.",
        "",
        "## Paired OOF deltas by satellite",
        "",
        "Negative values mean exp038 is better.",
        "",
        "| Comparator | Overall | GOES | Himawari | Meteosat | 95% location-bootstrap CI |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for comparator in [e for e in EXPERIMENTS if e != REFERENCE]:
        values = satellite[satellite["comparator"] == comparator].set_index("group")
        b = boot.loc[comparator]
        lines.append(
            f"| {comparator} | {f6(float(overall.loc[comparator, 'paired_delta_reference_minus_comparator']))} "
            f"| {f6(float(values.loc['goes', 'paired_delta_reference_minus_comparator']))} "
            f"| {f6(float(values.loc['himawari', 'paired_delta_reference_minus_comparator']))} "
            f"| {f6(float(values.loc['meteosat', 'paired_delta_reference_minus_comparator']))} "
            f"| [{f6(float(b['ci_2p5']))}, {f6(float(b['ci_97p5']))}] |"
        )
    lines += [
        "",
        "## Metadata reweight stress test",
        "",
        "| Scheme | exp011 | exp018 | exp035_no_dilation | exp038 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for scheme, row in reweight_pivot.iterrows():
        lines.append(
            f"| {scheme} | {f6(float(row['exp011']))} | {f6(float(row['exp018']))} "
            f"| {f6(float(row['exp035_no_dilation']))} | {f6(float(row['exp038']))} |"
        )
    lines += [
        "",
        "## Post-hoc CV/Public comparison",
        "",
        "This table may diagnose transfer but must not be treated as a new validation set.",
        "",
        "| Experiment | OOF | Public | Public−OOF | OOF Δ vs exp038 | Public Δ vs exp038 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in public_cv.iterrows():
        lines.append(
            f"| {row['experiment']} | {f6(float(row['oof_tile_rmse']))} | {f6(float(row['public_rmse']))} "
            f"| {f6(float(row['public_minus_oof']))} | {f6(float(row['oof_delta_vs_exp038']))} "
            f"| {f6(float(row['public_delta_vs_exp038']))} |"
        )
    lines += [
        "",
        "## Decision",
        "",
        "1. CV is reliable for the large strict gain exp038 vs exp011 and remains the primary gate.",
        "2. Differences around 0.003–0.007 are not stable enough to rank close models from a single aggregate OOF number.",
        "3. The next experiment should isolate the official tile-RMSE training objective on the exp038 architecture, using identical checkpoint initialization and continuation for control and metric-loss arms. It should not tune a global amplitude multiplier from the exp038 Public result.",
        "4. Promotion still requires fold 0 and fold 4 to improve in the same direction, followed by full five-fold OOF.",
        "",
        "Detailed subgroup tables are in the adjacent CSV files.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.bootstrap <= 0:
        raise ValueError("--bootstrap must be positive")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    oof, oof_paths = load_oof()
    train = metadata(TRAIN_CSV)
    evaluation = metadata(EVAL_CSV)

    paired = pd.DataFrame(paired_group_rows(oof))
    rng = np.random.default_rng(args.seed)
    bootstrap = pd.DataFrame(
        [
            stratified_location_bootstrap(oof, comparator, args.bootstrap, rng)
            for comparator in EXPERIMENTS
            if comparator != REFERENCE
        ]
    )
    reweighted = pd.DataFrame(metadata_reweight_rows(oof, train, evaluation))
    eval_prediction_rows, eval_paths = evaluation_prediction_rows()
    eval_predictions = pd.DataFrame(eval_prediction_rows)
    public_cv = pd.DataFrame(public_cv_rows(oof))

    tables = {
        "paired_deltas.csv": paired,
        "location_bootstrap.csv": bootstrap,
        "metadata_reweighted_scores.csv": reweighted,
        "evaluation_prediction_summary.csv": eval_predictions,
        "public_cv_comparison.csv": public_cv,
    }
    for name, frame in tables.items():
        frame.to_csv(out_dir / name, index=False)

    write_report(
        out_dir / "TRANSFER_AUDIT.md",
        paired,
        bootstrap,
        reweighted,
        eval_predictions,
        public_cv,
    )
    input_paths = {**{f"oof_{k}": v for k, v in oof_paths.items()}, **{f"eval_{k}": v for k, v in eval_paths.items()}, "train_csv": TRAIN_CSV, "evaluation_csv": EVAL_CSV}
    summary = {
        "reference": REFERENCE,
        "experiments": list(EXPERIMENTS),
        "n_oof": len(oof),
        "n_evaluation": len(evaluation),
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "public_scores_are_post_hoc_only": True,
        "inputs": {
            key: {"path": str(path.relative_to(PROJECT_DIR)), "sha256": sha256(path)}
            for key, path in input_paths.items()
        },
        "outputs": list(tables) + ["TRANSFER_AUDIT.md", "summary.json"],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote strict transfer audit to {out_dir}")


if __name__ == "__main__":
    main()
