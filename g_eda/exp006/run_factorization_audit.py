#!/usr/bin/env python3
"""P0 EDA: exact field-factorization and metric-aggregation audit.

This job consumes the reusable OOF field caches produced by g_eda/exp003 for
exp016/exp017/exp018.  Those experiments use successor-row context and are therefore
AMBER screening artifacts.  Nothing written by this script is a submission artifact.

Analyses
--------
1. Exact additive decomposition per tile::

       MSE = (mean(pred) - mean(target))**2
             + mean(((pred-mean(pred)) - (target-mean(target)))**2)

2. Multiplicative ``mean * normalized_shape`` decomposition, including its interaction.
3. True-mean scaling versus the non-negative per-tile L2-optimal scalar.
4. A 3x3 amount-source x shape-source cross-swap among exp016/017/018.
5. Tile-mean RMSE versus global-pooled RMSE scale stress.
6. Observation-count audit and a strict-row-only additive reference from exp011 CSV.

Target-dependent oracle quantities are diagnostics only.  They must never be used to
construct evaluation predictions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp003"
DEFAULT_OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp006"
TRAIN_CSV = PROJECT_DIR / "data" / "train_dataset" / "train_dataset.csv"
STRICT_SAMPLE_CSV = PROJECT_DIR / "outputs" / "analysis" / "exp011" / "oof_sample_metrics.csv"
WEIGHT_JSON = CACHE_DIR / "recommended_weights.json"
EXPERIMENTS = ("exp016", "exp017", "exp018")
PIXELS = 41 * 41
EPS = 1e-8
CACHE_KEYS = ("unique_id", "pred", "target", "satellite", "fold")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def safe_mse(values: np.ndarray) -> np.ndarray:
    """Clamp only floating-point cancellation around zero, not genuine quantities."""
    return np.maximum(np.asarray(values, dtype=np.float64), 0.0)


def score_from_mse(mse: np.ndarray, mask: np.ndarray | None = None) -> tuple[float, float]:
    values = safe_mse(mse if mask is None else mse[mask])
    if values.size == 0:
        return math.nan, math.nan
    return float(np.sqrt(values).mean()), float(np.sqrt(values.mean()))


def field_moments(pred: np.ndarray, target: np.ndarray, yy: np.ndarray) -> dict[str, np.ndarray]:
    pixels = int(pred.shape[1] * pred.shape[2])
    pmean = pred.mean(axis=(1, 2), dtype=np.float64)
    pp = np.einsum("nij,nij->n", pred, pred, dtype=np.float64, optimize=True) / pixels
    py = np.einsum("nij,nij->n", pred, target, dtype=np.float64, optimize=True) / pixels
    total_mse = safe_mse(pp - 2.0 * py + yy)
    return {"mean": pmean, "pp": pp, "py": py, "total_mse": total_mse}


def exact_factorization(
    pred_moments: dict[str, np.ndarray], ymean: np.ndarray, yy: np.ndarray
) -> dict[str, np.ndarray]:
    pmean = pred_moments["mean"]
    pp = pred_moments["pp"]
    py = pred_moments["py"]
    total = pred_moments["total_mse"]

    bias = pmean - ymean
    mean_mse = np.square(bias)
    centered_mse = safe_mse(total - mean_mse)
    additive_error = float(np.max(np.abs(total - (mean_mse + centered_mse))))
    if additive_error > 5e-5:
        raise AssertionError(f"additive decomposition drifted by {additive_error}")

    valid = (pmean > EPS) & (ymean > EPS)
    pure_mean_term = np.full_like(total, np.nan)
    shape_term = np.full_like(total, np.nan)
    scale_shape_interaction_term = np.full_like(total, np.nan)
    shape_interaction_signed_cross_term = np.full_like(total, np.nan)
    dm = pmean[valid] - ymean[valid]
    pmean_v = pmean[valid]
    ymean_v = ymean[valid]
    pp_v = pp[valid]
    py_v = py[valid]
    yy_v = yy[valid]

    # pred = pmean*shat; target = ymean*shape, with mean(shat)=mean(shape)=1.
    # pred-target = A + S + I, where A=dm, S=ymean*(shat-shape),
    # I=dm*(shat-1). A is orthogonal to S and I; S and I retain a cross term.
    pure_mean_term[valid] = np.square(dm)
    shape_term[valid] = (
        np.square(ymean_v) * pp_v / np.square(pmean_v)
        - 2.0 * ymean_v * py_v / pmean_v
        + yy_v
    )
    scale_shape_interaction_term[valid] = np.square(dm) * (
        pp_v / np.square(pmean_v) - 1.0
    )
    shape_interaction_signed_cross_term[valid] = (
        2.0 * dm * ymean_v * pp_v / np.square(pmean_v)
        - 2.0 * dm * py_v / pmean_v
    )
    multiplicative_error = float(
        np.max(np.abs(
            total[valid] - pure_mean_term[valid] - shape_term[valid]
            - scale_shape_interaction_term[valid] - shape_interaction_signed_cross_term[valid]
        ))
    )
    if multiplicative_error > 2e-4:
        raise AssertionError(f"multiplicative decomposition drifted by {multiplicative_error}")

    mean_scale = np.zeros_like(total)
    mean_scale_valid = pmean > EPS
    mean_scale[mean_scale_valid] = ymean[mean_scale_valid] / pmean[mean_scale_valid]
    mean_scale_mse = np.empty_like(total)
    mean_scale_mse[mean_scale_valid] = safe_mse(
        np.square(mean_scale[mean_scale_valid]) * pp[mean_scale_valid]
        - 2.0 * mean_scale[mean_scale_valid] * py[mean_scale_valid]
        + yy[mean_scale_valid]
    )
    # Match the historical amount_swap fallback: a flat true-mean field when pred mean is zero.
    mean_scale_mse[~mean_scale_valid] = safe_mse(
        yy[~mean_scale_valid] - np.square(ymean[~mean_scale_valid])
    )

    optimal_scale = np.zeros_like(total)
    # pp has squared precipitation units, so compare it with EPS**2 rather than EPS.
    optimal_valid = pp > EPS * EPS
    optimal_scale[optimal_valid] = np.maximum(py[optimal_valid] / pp[optimal_valid], 0.0)
    optimal_scale_mse = safe_mse(
        np.square(optimal_scale) * pp - 2.0 * optimal_scale * py + yy
    )
    if np.any(optimal_scale_mse > mean_scale_mse + 1e-5):
        worst = float(np.max(optimal_scale_mse - mean_scale_mse))
        raise AssertionError(f"L2-optimal scalar lost to mean scalar by {worst}")

    total_norm = np.sqrt(total)
    mean_norm = np.abs(bias)
    centered_norm = np.sqrt(centered_mse)
    mean_shapley = 0.5 * (mean_norm + total_norm - centered_norm)
    centered_shapley = 0.5 * (centered_norm + total_norm - mean_norm)
    if not np.allclose(mean_shapley + centered_shapley, total_norm, atol=1e-8):
        raise AssertionError("tile-norm Shapley contributions do not sum to the tile norm")

    return {
        **pred_moments,
        "bias": bias,
        "mean_mse": mean_mse,
        "centered_mse": centered_mse,
        "mean_shapley": mean_shapley,
        "centered_shapley": centered_shapley,
        "multiplicative_valid": valid,
        "multiplicative_pure_mean_mse": pure_mean_term,
        "multiplicative_shape_mse": shape_term,
        "multiplicative_scale_shape_interaction_mse": scale_shape_interaction_term,
        "multiplicative_shape_interaction_signed_cross_term": shape_interaction_signed_cross_term,
        "mean_scale": mean_scale,
        "mean_scale_mse": mean_scale_mse,
        "optimal_scale": optimal_scale,
        "optimal_scale_mse": optimal_scale_mse,
    }


def amount_bins(ymean: np.ndarray) -> np.ndarray:
    labels = np.full(ymean.shape, "extreme_ge_1", dtype="<U20")
    labels[ymean <= EPS] = "dry"
    labels[(ymean > EPS) & (ymean < 0.1)] = "light_0_0.1"
    labels[(ymean >= 0.1) & (ymean < 0.5)] = "moderate_0.1_0.5"
    labels[(ymean >= 0.5) & (ymean < 1.0)] = "heavy_0.5_1"
    return labels


def wet_fraction_bins(wet_fraction: np.ndarray) -> np.ndarray:
    labels = np.full(wet_fraction.shape, "wet_gt_0.5", dtype="<U20")
    labels[wet_fraction <= EPS] = "dry"
    labels[(wet_fraction > EPS) & (wet_fraction <= 0.05)] = "wet_0_0.05"
    labels[(wet_fraction > 0.05) & (wet_fraction <= 0.25)] = "wet_0.05_0.25"
    labels[(wet_fraction > 0.25) & (wet_fraction <= 0.5)] = "wet_0.25_0.5"
    return labels


def categorical_groups(name: str, values: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    return [(name, str(value), values == value) for value in sorted(np.unique(values).tolist())]


def build_groups(bundle: dict[str, object], ymean: np.ndarray, wet_fraction: np.ndarray) -> list[tuple[str, str, np.ndarray]]:
    n = len(ymean)
    groups: list[tuple[str, str, np.ndarray]] = [("global", "all", np.ones(n, dtype=bool))]
    groups += categorical_groups("fold", bundle["fold"])
    groups += categorical_groups("satellite", bundle["satellite"])
    groups += categorical_groups("location", bundle["location"])
    groups += categorical_groups(
        "own_row_observation_count", bundle["own_row_observation_count"]
    )
    amount = amount_bins(ymean)
    wet_bins = wet_fraction_bins(wet_fraction)
    groups += categorical_groups("target_amount_bin", amount)
    groups += categorical_groups("target_wet_fraction_bin", wet_bins)
    sat_amount = np.char.add(np.char.add(bundle["satellite"].astype(str), ":"), amount)
    groups += categorical_groups("satellite_amount_bin", sat_amount)
    return groups


def core_groups(groups: list[tuple[str, str, np.ndarray]]) -> list[tuple[str, str, np.ndarray]]:
    keep = {"global", "fold", "satellite"}
    return [group for group in groups if group[0] in keep]


def load_bundle(max_tiles: int | None) -> dict[str, object]:
    cache_arrays: dict[str, dict[str, np.ndarray]] = {}
    cache_schema: dict[str, dict[str, dict[str, object]]] = {}
    ref_full_ids: np.ndarray | None = None
    ref_ids: np.ndarray | None = None
    cache_paths: list[Path] = []
    for experiment in EXPERIMENTS:
        path = CACHE_DIR / f"{experiment}_oof_pred.npz"
        cache_paths.append(path)
        if not path.exists():
            raise FileNotFoundError(f"missing OOF cache: {path}; run g_eda/exp003 first")
        with np.load(path, allow_pickle=False) as cache:
            missing_keys = sorted(set(CACHE_KEYS) - set(cache.files))
            if missing_keys:
                raise ValueError(f"{experiment}: cache keys missing: {missing_keys}")
            cache_schema[experiment] = {
                key: {"shape": list(cache[key].shape), "dtype": str(cache[key].dtype)}
                for key in CACHE_KEYS
            }
            full_ids = np.asarray(cache["unique_id"])
            n = len(full_ids)
            if full_ids.ndim != 1 or len(np.unique(full_ids)) != n:
                raise ValueError(f"{experiment}: unique_id must be a unique 1-D key")
            if cache["pred"].shape != (n, 41, 41) or cache["target"].shape != (n, 41, 41):
                raise ValueError(f"{experiment}: pred/target must have shape (N, 41, 41)")
            if cache["satellite"].shape != (n,) or cache["fold"].shape != (n,):
                raise ValueError(f"{experiment}: satellite/fold must have shape (N,)")
            pred = cache["pred"].astype(np.float32)
            target = cache["target"].astype(np.float32)
            if not np.isfinite(pred).all() or not np.isfinite(target).all():
                raise ValueError(f"{experiment}: pred/target contains non-finite values")
            if float(pred.min()) < -1e-6 or float(target.min()) < -1e-6:
                raise ValueError(f"{experiment}: pred/target contains negative precipitation")
            order = np.argsort(full_ids)
            sorted_full_ids = full_ids[order]
            arrays = {
                "unique_id": sorted_full_ids,
                "pred": pred[order],
                "target": target[order],
                "satellite": cache["satellite"][order],
                "fold": cache["fold"][order],
            }
        if ref_full_ids is None:
            ref_full_ids = sorted_full_ids
        elif not np.array_equal(ref_full_ids, sorted_full_ids):
            raise ValueError(f"{experiment}: full unique_id set differs from the reference")
        if max_tiles is not None:
            arrays = {key: value[:max_tiles] for key, value in arrays.items()}
        if ref_ids is None:
            ref_ids = arrays["unique_id"]
        elif not np.array_equal(ref_ids, arrays["unique_id"]):
            raise ValueError(f"{experiment}: unique_id ordering differs from the reference")
        cache_arrays[experiment] = arrays

    reference = cache_arrays[EXPERIMENTS[0]]
    for experiment in EXPERIMENTS[1:]:
        current = cache_arrays[experiment]
        for key in ("target", "satellite", "fold"):
            if not np.array_equal(reference[key], current[key]):
                raise ValueError(f"{experiment}: {key} differs from {EXPERIMENTS[0]}")

    metadata: dict[str, tuple[str, str, int]] = {}
    with TRAIN_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["unique_id"]
            if uid in metadata:
                raise ValueError(f"train CSV contains duplicate unique_id: {uid}")
            observations = ast.literal_eval(row["last_30_minutes_observation_filename"])
            if not isinstance(observations, (list, tuple)) or not all(
                isinstance(value, str) for value in observations
            ):
                raise ValueError(f"{uid}: observation filename field is not a string list")
            metadata[uid] = (
                row["name_location"], row["satellite_target"], len(observations)
            )
    assert ref_full_ids is not None
    full_cache_ids = {str(uid) for uid in ref_full_ids.tolist()}
    train_ids = set(metadata)
    if max_tiles is None and full_cache_ids != train_ids:
        missing = sorted(train_ids - full_cache_ids)
        extra = sorted(full_cache_ids - train_ids)
        raise ValueError(
            "full OOF/train ID sets differ: "
            f"missing_from_cache={len(missing)}, extra_in_cache={len(extra)}"
        )

    ids = reference["unique_id"]
    missing_metadata = [uid for uid in ids if str(uid) not in metadata]
    if missing_metadata:
        raise KeyError(
            f"{len(missing_metadata)} OOF IDs are absent from train CSV; "
            f"first={missing_metadata[0]}"
        )
    location = np.asarray([metadata[str(uid)][0] for uid in ids])
    csv_satellite = np.asarray([metadata[str(uid)][1] for uid in ids])
    own_row_observation_count = np.asarray(
        [metadata[str(uid)][2] for uid in ids], dtype=np.int8
    )
    if not np.array_equal(reference["satellite"].astype(str), csv_satellite.astype(str)):
        raise ValueError("cache satellite labels differ from train CSV")
    for name in np.unique(location):
        folds = np.unique(reference["fold"][location == name])
        if len(folds) != 1:
            raise ValueError(f"location {name} maps to multiple folds: {folds.tolist()}")

    return {
        "unique_id": ids,
        "target": reference["target"],
        "satellite": reference["satellite"],
        "fold": reference["fold"],
        "location": location,
        "own_row_observation_count": own_row_observation_count,
        "predictions": {exp: cache_arrays[exp]["pred"] for exp in EXPERIMENTS},
        "cache_paths": cache_paths,
        "data_contract": {
            "cache_schema": cache_schema,
            "full_cache_rows": int(len(ref_full_ids)),
            "train_unique_rows": int(len(metadata)),
            "analysis_rows": int(len(ids)),
            "full_id_set_equals_train": bool(full_cache_ids == train_ids),
        },
    }


def factorization_summary_rows(
    experiment: str,
    diagnostics: dict[str, np.ndarray],
    groups: list[tuple[str, str, np.ndarray]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for group_type, group_value, mask in groups:
        n = int(mask.sum())
        if n == 0:
            continue
        total_t, total_g = score_from_mse(diagnostics["total_mse"], mask)
        centered_t, centered_g = score_from_mse(diagnostics["centered_mse"], mask)
        mean_scale_t, mean_scale_g = score_from_mse(diagnostics["mean_scale_mse"], mask)
        optimal_t, optimal_g = score_from_mse(diagnostics["optimal_scale_mse"], mask)
        denominator_t = total_t - optimal_t
        denominator_g = total_g - optimal_g
        valid = mask & diagnostics["multiplicative_valid"]
        valid_n = int(valid.sum())
        total_sse = float(diagnostics["total_mse"][mask].sum())
        valid_total_sse = float(diagnostics["total_mse"][valid].sum()) if valid_n else math.nan
        row = {
            "experiment": experiment,
            "risk": "amber_successor_screening",
            "source_context": "successor_row_oof_cache",
            "contains_target_oracle": True,
            "deployable_at_evaluation": False,
            "group_type": group_type,
            "group_value": group_value,
            "samples": n,
            "tile_rmse": total_t,
            "global_rmse": total_g,
            "mean_abs_bias": float(np.abs(diagnostics["bias"][mask]).mean()),
            "mean_bias_rmse": float(np.sqrt(diagnostics["mean_mse"][mask].mean())),
            "centered_tile_rmse": centered_t,
            "centered_global_rmse": centered_g,
            "mean_sse_share": float(diagnostics["mean_mse"][mask].sum() / max(total_sse, EPS)),
            "centered_sse_share": float(diagnostics["centered_mse"][mask].sum() / max(total_sse, EPS)),
            "mean_tile_shapley": float(diagnostics["mean_shapley"][mask].mean()),
            "centered_tile_shapley": float(diagnostics["centered_shapley"][mask].mean()),
            "mean_tile_shapley_share": float(
                diagnostics["mean_shapley"][mask].sum()
                / max(float(np.sqrt(diagnostics["total_mse"][mask]).sum()), EPS)
            ),
            "multiplicative_valid_samples": valid_n,
            "multiplicative_pure_mean_sse_share": (
                float(np.nansum(diagnostics["multiplicative_pure_mean_mse"][valid]) / valid_total_sse)
                if valid_n and valid_total_sse > EPS else math.nan
            ),
            "multiplicative_shape_sse_share": (
                float(np.nansum(diagnostics["multiplicative_shape_mse"][valid]) / valid_total_sse)
                if valid_n and valid_total_sse > EPS else math.nan
            ),
            "multiplicative_scale_shape_interaction_sse_share": (
                float(np.nansum(diagnostics["multiplicative_scale_shape_interaction_mse"][valid]) / valid_total_sse)
                if valid_n and valid_total_sse > EPS else math.nan
            ),
            "multiplicative_shape_interaction_signed_cross_term_share": (
                float(np.nansum(
                    diagnostics["multiplicative_shape_interaction_signed_cross_term"][valid]
                ) / valid_total_sse)
                if valid_n and valid_total_sse > EPS else math.nan
            ),
            "true_mean_scale_tile_rmse": mean_scale_t,
            "true_mean_scale_global_rmse": mean_scale_g,
            "optimal_scale_tile_rmse": optimal_t,
            "optimal_scale_global_rmse": optimal_g,
            "true_mean_scale_tile_improvement": total_t - mean_scale_t,
            "optimal_scale_tile_improvement": total_t - optimal_t,
            "true_mean_scale_tile_recovery": (
                (total_t - mean_scale_t) / denominator_t if denominator_t > EPS else math.nan
            ),
            "true_mean_scale_global_recovery": (
                (total_g - mean_scale_g) / denominator_g if denominator_g > EPS else math.nan
            ),
            "median_true_mean_scale": float(np.median(diagnostics["mean_scale"][mask])),
            "median_optimal_scale": float(np.median(diagnostics["optimal_scale"][mask])),
        }
        rows.append(row)
    return rows


def cross_swap_mse(
    amount: dict[str, np.ndarray], shape: dict[str, np.ndarray], ymean: np.ndarray, yy: np.ndarray
) -> np.ndarray:
    amount_mean = amount["mean"]
    shape_mean = shape["mean"]
    valid = shape_mean > EPS
    ratio = np.zeros_like(shape_mean)
    ratio[valid] = amount_mean[valid] / shape_mean[valid]
    mse = np.empty_like(shape_mean)
    mse[valid] = safe_mse(
        np.square(ratio[valid]) * shape["pp"][valid]
        - 2.0 * ratio[valid] * shape["py"][valid]
        + yy[valid]
    )
    # A shape with zero mean carries no spatial information; preserve the amount as a flat field.
    mse[~valid] = safe_mse(
        yy[~valid] - 2.0 * amount_mean[~valid] * ymean[~valid]
        + np.square(amount_mean[~valid])
    )
    return mse


def run_cross_swap(
    moments: dict[str, dict[str, np.ndarray]],
    ymean: np.ndarray,
    yy: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
) -> tuple[
    list[dict[str, object]],
    dict[str, object],
    dict[tuple[str, str], np.ndarray],
    list[dict[str, object]],
]:
    rows: list[dict[str, object]] = []
    fields: dict[tuple[str, str], np.ndarray] = {}
    for amount_exp in EXPERIMENTS:
        for shape_exp in EXPERIMENTS:
            mse = cross_swap_mse(moments[amount_exp], moments[shape_exp], ymean, yy)
            fields[(amount_exp, shape_exp)] = mse
            for group_type, group_value, mask in core_groups(groups):
                tile_score, global_score = score_from_mse(mse, mask)
                rows.append({
                    "risk": "amber_successor_screening",
                    "source_context": "final_prediction_tile_mean_x_normalized_field_shape",
                    "target_oracle": False,
                    "deployable_at_evaluation": False,
                    "amount_experiment": amount_exp,
                    "shape_experiment": shape_exp,
                    "off_diagonal": amount_exp != shape_exp,
                    "group_type": group_type,
                    "group_value": group_value,
                    "samples": int(mask.sum()),
                    "tile_rmse": tile_score,
                    "global_rmse": global_score,
                })

    global_rows = [row for row in rows if row["group_type"] == "global"]
    best_tile = min(global_rows, key=lambda row: float(row["tile_rmse"]))
    best_global = min(global_rows, key=lambda row: float(row["global_rmse"]))
    diagonal = [row for row in global_rows if not bool(row["off_diagonal"])]
    off_diagonal = [row for row in global_rows if bool(row["off_diagonal"])]
    best_diagonal_tile = min(diagonal, key=lambda row: float(row["tile_rmse"]))
    best_diagonal_global = min(diagonal, key=lambda row: float(row["global_rmse"]))
    best_off_diagonal_tile = min(off_diagonal, key=lambda row: float(row["tile_rmse"]))
    best_off_diagonal_global = min(off_diagonal, key=lambda row: float(row["global_rmse"]))

    fold_masks = {
        int(value): mask for group_type, value, mask in groups if group_type == "fold"
    }

    def post_selection_fold_wins(
        candidate: dict[str, object], metric: str, baseline: dict[str, object]
    ) -> int:
        wins = 0
        key = (str(candidate["amount_experiment"]), str(candidate["shape_experiment"]))
        base_key = (str(baseline["amount_experiment"]), str(baseline["shape_experiment"]))
        for fold, mask in sorted(fold_masks.items()):
            cand_score = score_from_mse(fields[key], mask)[0 if metric == "tile_rmse" else 1]
            base_score = score_from_mse(fields[base_key], mask)[0 if metric == "tile_rmse" else 1]
            if cand_score <= base_score - 0.003:
                wins += 1
        return wins

    diagonal_keys = [(exp, exp) for exp in EXPERIMENTS]
    off_diagonal_keys = [
        (amount_exp, shape_exp)
        for amount_exp in EXPERIMENTS
        for shape_exp in EXPERIMENTS
        if amount_exp != shape_exp
    ]
    lofo_rows: list[dict[str, object]] = []
    for metric, score_index in (("tile_rmse", 0), ("global_rmse", 1)):
        for heldout_fold, heldout_mask in sorted(fold_masks.items()):
            training_mask = ~heldout_mask

            def training_score(key: tuple[str, str]) -> float:
                return score_from_mse(fields[key], training_mask)[score_index]

            candidate_key = min(off_diagonal_keys, key=training_score)
            baseline_key = min(diagonal_keys, key=training_score)
            candidate_score = score_from_mse(fields[candidate_key], heldout_mask)[score_index]
            baseline_score = score_from_mse(fields[baseline_key], heldout_mask)[score_index]
            delta = candidate_score - baseline_score
            lofo_rows.append({
                "risk": "amber_successor_screening",
                "selection_scheme": "leave_one_fold_out",
                "source_context": "final_prediction_tile_mean_x_normalized_field_shape",
                "target_oracle": False,
                "deployable_at_evaluation": False,
                "metric": metric,
                "heldout_fold": heldout_fold,
                "selected_amount_experiment": candidate_key[0],
                "selected_shape_experiment": candidate_key[1],
                "baseline_amount_experiment": baseline_key[0],
                "baseline_shape_experiment": baseline_key[1],
                "candidate_score": candidate_score,
                "baseline_score": baseline_score,
                "delta": delta,
                "win_ge_0.003": bool(delta <= -0.003),
            })

    summary = {
        "best_tile": best_tile,
        "best_global": best_global,
        "best_diagonal_tile": best_diagonal_tile,
        "best_diagonal_global": best_diagonal_global,
        "best_off_diagonal_tile": best_off_diagonal_tile,
        "best_off_diagonal_global": best_off_diagonal_global,
        "best_off_diagonal_tile_vs_diagonal_delta": (
            float(best_off_diagonal_tile["tile_rmse"]) - float(best_diagonal_tile["tile_rmse"])
        ),
        "best_off_diagonal_global_vs_diagonal_delta": (
            float(best_off_diagonal_global["global_rmse"]) - float(best_diagonal_global["global_rmse"])
        ),
        "post_selection_best_off_diagonal_tile_fold_wins_ge_0.003": post_selection_fold_wins(
            best_off_diagonal_tile, "tile_rmse", best_diagonal_tile
        ),
        "post_selection_best_off_diagonal_global_fold_wins_ge_0.003": post_selection_fold_wins(
            best_off_diagonal_global, "global_rmse", best_diagonal_global
        ),
        "lofo_off_diagonal_tile_fold_wins_ge_0.003": sum(
            int(row["win_ge_0.003"]) for row in lofo_rows if row["metric"] == "tile_rmse"
        ),
        "lofo_off_diagonal_global_fold_wins_ge_0.003": sum(
            int(row["win_ge_0.003"]) for row in lofo_rows if row["metric"] == "global_rmse"
        ),
    }
    return rows, summary, fields, lofo_rows


def load_blend_weights() -> tuple[dict[str, float], str]:
    default = {"exp016": 0.25, "exp017": 0.30, "exp018": 0.45}
    if not WEIGHT_JSON.exists():
        return default, "built_in_fallback"
    payload = json.loads(WEIGHT_JSON.read_text(encoding="utf-8"))
    record = payload.get("global_best", {})
    weights = {
        "exp016": float(record.get("w016", default["exp016"])),
        "exp017": float(record.get("w017", default["exp017"])),
        "exp018": float(record.get("w018", default["exp018"])),
    }
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("blend weights sum to zero")
    return {key: value / total for key, value in weights.items()}, str(WEIGHT_JSON)


def run_metric_stress(
    predictions: dict[str, np.ndarray],
    target: np.ndarray,
    yy: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
    scale_min: float,
    scale_max: float,
    scale_step: float,
) -> tuple[list[dict[str, object]], dict[str, object], np.ndarray, list[dict[str, object]]]:
    weights, weight_source = load_blend_weights()
    blend = np.zeros_like(target, dtype=np.float32)
    for experiment, weight in weights.items():
        blend += weight * predictions[experiment]
    blend_moments = field_moments(blend, target, yy)
    del blend

    scales = np.arange(scale_min, scale_max + scale_step / 2.0, scale_step)
    rows: list[dict[str, object]] = []
    stress_groups = core_groups(groups)
    mse_by_scale: dict[float, np.ndarray] = {}
    for raw_scale in scales:
        scale = float(round(float(raw_scale), 10))
        mse = safe_mse(
            scale * scale * blend_moments["pp"]
            - 2.0 * scale * blend_moments["py"]
            + yy
        )
        mse_by_scale[scale] = mse
        for group_type, group_value, mask in stress_groups:
            tile_score, global_score = score_from_mse(mse, mask)
            rows.append({
                "risk": "amber_successor_screening",
                "source_context": "recommended_oof_blend_common_scale",
                "target_oracle": False,
                "deployable_at_evaluation": False,
                "source": "global_oof_blend",
                "weight_source": weight_source,
                "weights": json.dumps(weights, sort_keys=True),
                "scale": scale,
                "group_type": group_type,
                "group_value": group_value,
                "samples": int(mask.sum()),
                "tile_rmse": tile_score,
                "global_rmse": global_score,
            })

    global_rows = [row for row in rows if row["group_type"] == "global"]
    tile_best = min(global_rows, key=lambda row: float(row["tile_rmse"]))
    pooled_best = min(global_rows, key=lambda row: float(row["global_rmse"]))
    tile_scale = float(tile_best["scale"])
    pooled_scale = float(pooled_best["scale"])
    tile_at_pooled = next(row for row in global_rows if float(row["scale"]) == pooled_scale)
    pooled_at_tile = next(row for row in global_rows if float(row["scale"]) == tile_scale)
    analytic_global_scale = float(
        max(blend_moments["py"].sum() / max(blend_moments["pp"].sum(), EPS), 0.0)
    )
    summary = {
        "weights": weights,
        "weight_source": weight_source,
        "tile_optimum": tile_best,
        "global_optimum": pooled_best,
        "analytic_global_scale": analytic_global_scale,
        "scale_gap": abs(tile_scale - pooled_scale),
        "tile_delta_tile_opt_minus_global_opt": float(tile_best["tile_rmse"]) - float(tile_at_pooled["tile_rmse"]),
        "global_delta_tile_opt_minus_global_opt": float(pooled_at_tile["global_rmse"]) - float(pooled_best["global_rmse"]),
        "material_reversal": (
            abs(tile_scale - pooled_scale) > 0.05
            and float(tile_best["tile_rmse"]) <= float(tile_at_pooled["tile_rmse"]) - 0.003
            and float(pooled_at_tile["global_rmse"]) >= float(pooled_best["global_rmse"]) + 0.005
        ),
    }

    tail_rows: list[dict[str, object]] = []
    sources = {exp: field_moments(predictions[exp], target, yy)["total_mse"] for exp in EXPERIMENTS}
    sources["global_oof_blend"] = blend_moments["total_mse"]
    for source, mse in sources.items():
        order = np.argsort(mse)[::-1]
        tile_norm = np.sqrt(mse)
        for fraction in (0.05, 0.10):
            count = max(1, int(math.ceil(len(mse) * fraction)))
            selected = order[:count]
            tail_rows.append({
                "risk": "amber_successor_screening",
                "source_context": "oof_error_tail_diagnostic",
                "target_oracle": False,
                "deployable_at_evaluation": False,
                "source": source,
                "top_fraction": fraction,
                "tiles": count,
                "tile_norm_mass_share": float(tile_norm[selected].sum() / max(tile_norm.sum(), EPS)),
                "sse_mass_share": float(mse[selected].sum() / max(mse.sum(), EPS)),
            })
    return rows, summary, blend_moments["total_mse"], tail_rows


def strict_additive_reference(
    ids: np.ndarray,
    groups: list[tuple[str, str, np.ndarray]],
    require_exact_ids: bool,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    if not STRICT_SAMPLE_CSV.exists():
        raise FileNotFoundError(f"strict reference missing: {STRICT_SAMPLE_CSV}")
    records: dict[str, dict[str, str]] = {}
    with STRICT_SAMPLE_CSV.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = row["unique_id"]
            if uid in records:
                raise ValueError(f"strict exp011 CSV contains duplicate unique_id: {uid}")
            records[uid] = row
    id_set = {str(uid) for uid in ids.tolist()}
    if require_exact_ids and id_set != set(records):
        raise ValueError(
            "strict exp011/cache ID sets differ: "
            f"missing_from_strict={len(id_set - set(records))}, "
            f"extra_in_strict={len(set(records) - id_set)}"
        )
    missing = [uid for uid in ids if str(uid) not in records]
    if missing:
        raise KeyError(f"strict exp011 CSV is missing {len(missing)} IDs; first={missing[0]}")
    tile = np.asarray([float(records[str(uid)]["tile_rmse"]) for uid in ids], dtype=np.float64)
    bias = np.asarray([float(records[str(uid)]["bias"]) for uid in ids], dtype=np.float64)
    if not np.isfinite(tile).all() or not np.isfinite(bias).all() or np.any(tile < 0):
        raise ValueError("strict exp011 CSV contains invalid tile_rmse/bias values")
    total_mse = np.square(tile)
    mean_mse = np.square(bias)
    centered_mse = safe_mse(total_mse - mean_mse)
    rows: list[dict[str, object]] = []
    for group_type, group_value, mask in groups:
        actual_t, actual_g = score_from_mse(total_mse, mask)
        centered_t, centered_g = score_from_mse(centered_mse, mask)
        total_sse = float(total_mse[mask].sum())
        rows.append({
            "experiment": "exp011",
            "risk": "strict_row_only_reference",
            "source_context": "strict_row_only_oof_summary",
            "target_oracle": True,
            "deployable_at_evaluation": False,
            "group_type": group_type,
            "group_value": group_value,
            "samples": int(mask.sum()),
            "tile_rmse": actual_t,
            "global_rmse": actual_g,
            "mean_abs_bias": float(np.abs(bias[mask]).mean()),
            "mean_bias_rmse": float(np.sqrt(mean_mse[mask].mean())),
            "centered_tile_rmse": centered_t,
            "centered_global_rmse": centered_g,
            "mean_sse_share": float(mean_mse[mask].sum() / max(total_sse, EPS)),
            "centered_sse_share": float(centered_mse[mask].sum() / max(total_sse, EPS)),
            "mean_bias_removal_tile_improvement": actual_t - centered_t,
            "mean_bias_removal_global_improvement": actual_g - centered_g,
        })
    return rows, {"total_mse": total_mse, "centered_mse": centered_mse, "bias": bias}


def availability_rows(
    bundle: dict[str, object],
    ymean: np.ndarray,
    wet_fraction: np.ndarray,
    strict: dict[str, np.ndarray],
    exploratory: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    satellites = bundle["satellite"]
    observations = bundle["own_row_observation_count"]
    rows: list[dict[str, object]] = []
    for satellite in ("all", *sorted(np.unique(satellites).tolist())):
        sat_mask = np.ones(len(ymean), dtype=bool) if satellite == "all" else satellites == satellite
        for count in sorted(np.unique(observations[sat_mask]).tolist()):
            mask = sat_mask & (observations == count)
            strict_t, strict_g = score_from_mse(strict["total_mse"], mask)
            strict_center_t, _ = score_from_mse(strict["centered_mse"], mask)
            exp_t, exp_g = score_from_mse(exploratory["total_mse"], mask)
            exp_center_t, _ = score_from_mse(exploratory["centered_mse"], mask)
            rows.append({
                "risk": "mixed_strict_and_amber_descriptive",
                "source_context": "own_train_csv_row_not_successor_model_context",
                "target_oracle": True,
                "deployable_at_evaluation": False,
                "satellite": satellite,
                "own_row_observation_count": int(count),
                "samples": int(mask.sum()),
                "target_mean": float(ymean[mask].mean()),
                "target_wet_fraction": float(wet_fraction[mask].mean()),
                "strict_exp011_tile_rmse": strict_t,
                "strict_exp011_global_rmse": strict_g,
                "strict_exp011_centered_tile_rmse": strict_center_t,
                "exploratory_exp018_tile_rmse": exp_t,
                "exploratory_exp018_global_rmse": exp_g,
                "exploratory_exp018_centered_tile_rmse": exp_center_t,
            })
    return rows


def paired_location_bootstrap(
    base_mse: np.ndarray,
    candidate_mse: np.ndarray,
    locations: np.ndarray,
    satellites: np.ndarray,
    reps: int,
    seed: int,
) -> dict[str, object]:
    unique_locations = np.asarray(sorted(np.unique(locations).tolist()))
    n_locations = len(unique_locations)
    base_tile_sum = np.empty(n_locations, dtype=np.float64)
    candidate_tile_sum = np.empty(n_locations, dtype=np.float64)
    base_sse_sum = np.empty(n_locations, dtype=np.float64)
    candidate_sse_sum = np.empty(n_locations, dtype=np.float64)
    counts = np.empty(n_locations, dtype=np.float64)
    location_satellite = np.empty(n_locations, dtype="<U16")
    for i, location in enumerate(unique_locations):
        mask = locations == location
        satellite_values = np.unique(satellites[mask])
        if len(satellite_values) != 1:
            raise ValueError(f"location {location} maps to multiple satellites: {satellite_values}")
        location_satellite[i] = satellite_values[0]
        counts[i] = mask.sum()
        base_tile_sum[i] = np.sqrt(base_mse[mask]).sum()
        candidate_tile_sum[i] = np.sqrt(candidate_mse[mask]).sum()
        base_sse_sum[i] = base_mse[mask].sum()
        candidate_sse_sum[i] = candidate_mse[mask].sum()

    rng = np.random.default_rng(seed)
    sampled_parts = []
    for satellite in sorted(np.unique(location_satellite).tolist()):
        stratum = np.flatnonzero(location_satellite == satellite)
        sampled_parts.append(rng.choice(stratum, size=(reps, len(stratum)), replace=True))
    indices = np.concatenate(sampled_parts, axis=1)
    sampled_counts = counts[indices].sum(axis=1)
    tile_delta = (
        candidate_tile_sum[indices].sum(axis=1) / sampled_counts
        - base_tile_sum[indices].sum(axis=1) / sampled_counts
    )
    global_delta = (
        np.sqrt(candidate_sse_sum[indices].sum(axis=1) / sampled_counts)
        - np.sqrt(base_sse_sum[indices].sum(axis=1) / sampled_counts)
    )
    base_tile, base_global = score_from_mse(base_mse)
    candidate_tile, candidate_global = score_from_mse(candidate_mse)
    return {
        "estimand": "row_weighted_delta_over_satellite_stratified_location_population",
        "bootstrap_seed": seed,
        "tile_delta_observed": candidate_tile - base_tile,
        "tile_delta_mean": float(tile_delta.mean()),
        "tile_delta_ci_low": float(np.quantile(tile_delta, 0.025)),
        "tile_delta_ci_high": float(np.quantile(tile_delta, 0.975)),
        "global_delta_observed": candidate_global - base_global,
        "global_delta_mean": float(global_delta.mean()),
        "global_delta_ci_low": float(np.quantile(global_delta, 0.025)),
        "global_delta_ci_high": float(np.quantile(global_delta, 0.975)),
        "satellite_strata": int(len(np.unique(location_satellite))),
    }


def bootstrap_rows(
    diagnostics: dict[str, dict[str, np.ndarray]],
    strict: dict[str, np.ndarray],
    locations: np.ndarray,
    satellites: np.ndarray,
    reps: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    comparisons: list[tuple[str, str, np.ndarray, np.ndarray]] = []
    for experiment, record in diagnostics.items():
        comparisons.append((experiment, "true_mean_scale_minus_actual", record["total_mse"], record["mean_scale_mse"]))
        comparisons.append((experiment, "optimal_scale_minus_actual", record["total_mse"], record["optimal_scale_mse"]))
        comparisons.append((experiment, "mean_bias_removed_minus_actual", record["total_mse"], record["centered_mse"]))
    comparisons.append(("exp011", "mean_bias_removed_minus_actual", strict["total_mse"], strict["centered_mse"]))
    for i, (experiment, comparison, base, candidate) in enumerate(comparisons):
        result = paired_location_bootstrap(
            base, candidate, locations, satellites, reps, seed + i
        )
        rows.append({
            "experiment": experiment,
            "comparison": comparison,
            "risk": "conditional_target_oracle_opportunity",
            "source_context": (
                "amber_successor_oof" if experiment in EXPERIMENTS
                else "strict_row_only_oof_summary"
            ),
            "target_oracle": True,
            "deployable_at_evaluation": False,
            "cluster_unit": "location_stratified_by_satellite",
            "locations": int(len(np.unique(locations))),
            "bootstrap_reps": reps,
            **result,
        })
    return rows


def write_tile_diagnostics(
    path: Path,
    bundle: dict[str, object],
    ymean: np.ndarray,
    wet_fraction: np.ndarray,
    diagnostics: dict[str, dict[str, np.ndarray]],
) -> None:
    fields = [
        "experiment", "risk", "source_context", "contains_target_oracle",
        "deployable_at_evaluation", "unique_id", "fold", "location", "satellite",
        "own_row_observation_count", "target_mean", "target_wet_fraction", "pred_mean",
        "tile_rmse", "mean_bias", "mean_mse", "centered_rmse", "centered_mse",
        "mean_tile_shapley", "centered_tile_shapley", "multiplicative_pure_mean_mse",
        "multiplicative_shape_mse", "multiplicative_scale_shape_interaction_mse",
        "multiplicative_shape_interaction_signed_cross_term", "true_mean_scale",
        "true_mean_scale_rmse", "optimal_scale", "optimal_scale_rmse",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for experiment in EXPERIMENTS:
            record = diagnostics[experiment]
            for i, unique_id in enumerate(bundle["unique_id"]):
                valid = bool(record["multiplicative_valid"][i])
                writer.writerow({
                    "experiment": experiment,
                    "risk": "amber_successor_screening",
                    "source_context": "successor_row_oof_cache",
                    "contains_target_oracle": True,
                    "deployable_at_evaluation": False,
                    "unique_id": unique_id,
                    "fold": int(bundle["fold"][i]),
                    "location": bundle["location"][i],
                    "satellite": bundle["satellite"][i],
                    "own_row_observation_count": int(
                        bundle["own_row_observation_count"][i]
                    ),
                    "target_mean": ymean[i],
                    "target_wet_fraction": wet_fraction[i],
                    "pred_mean": record["mean"][i],
                    "tile_rmse": math.sqrt(record["total_mse"][i]),
                    "mean_bias": record["bias"][i],
                    "mean_mse": record["mean_mse"][i],
                    "centered_rmse": math.sqrt(record["centered_mse"][i]),
                    "centered_mse": record["centered_mse"][i],
                    "mean_tile_shapley": record["mean_shapley"][i],
                    "centered_tile_shapley": record["centered_shapley"][i],
                    "multiplicative_pure_mean_mse": record["multiplicative_pure_mean_mse"][i] if valid else "",
                    "multiplicative_shape_mse": record["multiplicative_shape_mse"][i] if valid else "",
                    "multiplicative_scale_shape_interaction_mse": (
                        record["multiplicative_scale_shape_interaction_mse"][i] if valid else ""
                    ),
                    "multiplicative_shape_interaction_signed_cross_term": (
                        record["multiplicative_shape_interaction_signed_cross_term"][i]
                        if valid else ""
                    ),
                    "true_mean_scale": record["mean_scale"][i],
                    "true_mean_scale_rmse": math.sqrt(record["mean_scale_mse"][i]),
                    "optimal_scale": record["optimal_scale"][i],
                    "optimal_scale_rmse": math.sqrt(record["optimal_scale_mse"][i]),
                })


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path),
        "bytes": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": file_sha256(path),
    }


def script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def global_row(rows: Iterable[dict[str, object]], experiment: str) -> dict[str, object]:
    return next(
        row for row in rows
        if row.get("experiment") == experiment
        and row.get("group_type") == "global"
        and row.get("group_value") == "all"
    )


def render_report(
    factor_rows: list[dict[str, object]],
    strict_rows: list[dict[str, object]],
    cross_summary: dict[str, object],
    metric_summary: dict[str, object],
    availability: list[dict[str, object]],
    bootstrap: list[dict[str, object]],
    elapsed: float,
    max_tiles: int | None,
) -> str:
    lines = [
        "# g_eda/exp006: Exact factorization and metric stress audit",
        "",
        f"- tiles: {factor_rows[0]['samples'] if factor_rows else 0}",
        f"- elapsed seconds: {elapsed:.1f}",
        f"- max_tiles: {max_tiles if max_tiles is not None else 'all'}",
        "- exp016/017/018 risk: **amber successor-context screening only**",
        "- exp011 additive reference: **strict row-only**",
        "- target-dependent oracle scales: **diagnostic only; never usable at evaluation**",
        "",
        "## Exact decomposition",
        "",
        "| model | risk | tile | global | mean-bias removed tile | mean SSE share | true-mean scale | L2 scale oracle | recovery |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment in EXPERIMENTS:
        row = global_row(factor_rows, experiment)
        lines.append(
            f"| {experiment} | amber | {row['tile_rmse']:.6f} | {row['global_rmse']:.6f} | "
            f"{row['centered_tile_rmse']:.6f} | {row['mean_sse_share']:.1%} | "
            f"{row['true_mean_scale_tile_rmse']:.6f} | {row['optimal_scale_tile_rmse']:.6f} | "
            f"{row['true_mean_scale_tile_recovery']:.1%} |"
        )
    strict = global_row(strict_rows, "exp011")
    lines.append(
        f"| exp011 | strict | {strict['tile_rmse']:.6f} | {strict['global_rmse']:.6f} | "
        f"{strict['centered_tile_rmse']:.6f} | {strict['mean_sse_share']:.1%} | n/a | n/a | n/a |"
    )
    lines += [
        "",
        "Reading: additive mean bias is not the same as multiplicative field-amplitude error.",
        "The true-mean scale changes the centered spatial residual amplitude as well.",
        "Mean-bias removal is an orthogonal diagnostic and can create negative pixels; it is not",
        "a submission-ready post-processing rule. Multiplicative terms use positive-mean tiles only.",
        "",
        "## Metric aggregation stress",
        "",
        f"- tile-optimal scale: **{metric_summary['tile_optimum']['scale']:.3f}** "
        f"(tile {metric_summary['tile_optimum']['tile_rmse']:.6f})",
        f"- global-optimal grid scale: **{metric_summary['global_optimum']['scale']:.3f}** "
        f"(global {metric_summary['global_optimum']['global_rmse']:.6f})",
        f"- analytic global scale: {metric_summary['analytic_global_scale']:.6f}",
        f"- scale gap: {metric_summary['scale_gap']:.3f}",
        f"- material ranking reversal: **{metric_summary['material_reversal']}**",
        "",
        "Do not freeze calibration or a direct serving loss until the organizer confirms whether",
        "the hidden evaluator pools all pixels or averages per-file norms.",
        "",
        "## Final-prediction amount component x normalized-field shape cross-swap",
        "",
        f"- tile best: amount={cross_summary['best_tile']['amount_experiment']}, "
        f"shape={cross_summary['best_tile']['shape_experiment']}, "
        f"score={cross_summary['best_tile']['tile_rmse']:.6f}",
        f"- global best: amount={cross_summary['best_global']['amount_experiment']}, "
        f"shape={cross_summary['best_global']['shape_experiment']}, "
        f"score={cross_summary['best_global']['global_rmse']:.6f}",
        f"- best off-diagonal tile delta versus best diagonal: "
        f"{cross_summary['best_off_diagonal_tile_vs_diagonal_delta']:+.6f}",
        f"- best off-diagonal global delta versus best diagonal: "
        f"{cross_summary['best_off_diagonal_global_vs_diagonal_delta']:+.6f}",
        f"- leave-one-fold-out off-diagonal wins >=0.003 (tile/global): "
        f"{cross_summary['lofo_off_diagonal_tile_fold_wins_ge_0.003']}/5 / "
        f"{cross_summary['lofo_off_diagonal_global_fold_wins_ge_0.003']}/5",
        "- full-OOF best-pair fold counts are post-selection diagnostics and are not an",
        "  independent replication result.",
        "",
        "## Input availability",
        "",
        "| own-row observations | samples |",
        "| ---: | ---: |",
    ]
    for row in availability:
        if row["satellite"] == "all":
            lines.append(f"| {row['own_row_observation_count']} | {row['samples']} |")
    lines += [
        "",
        "The availability count belongs to the CSV row itself, not the actual successor-context",
        "frames consumed by exp018. This table is descriptive; target amount and location are",
        "confounders. Use matched or outer-location analysis before claiming a missingness effect.",
        "",
        "## Location-cluster bootstrap",
        "",
        "| model | comparison | observed tile delta [95% CI] | observed global delta [95% CI] |",
        "| --- | --- | --- | --- |",
    ]
    for row in bootstrap:
        lines.append(
            f"| {row['experiment']} | {row['comparison']} | "
            f"{row['tile_delta_observed']:+.5f} "
            f"[{row['tile_delta_ci_low']:+.5f}, {row['tile_delta_ci_high']:+.5f}] | "
            f"{row['global_delta_observed']:+.5f} "
            f"[{row['global_delta_ci_low']:+.5f}, {row['global_delta_ci_high']:+.5f}] |"
        )
    lines += [
        "",
        "These intervals quantify conditional target-oracle opportunity across sampled locations;",
        "they are not confidence intervals for a deployable evaluation-time intervention.",
        "",
        "## Decision rules",
        "",
        "- Promote factorization to a training A/B only if true-mean scaling recovers at least 75%",
        "  of the L2-scale oracle gain in >=4/5 folds, then reproduce on strict-row-only OOF.",
        "- Promote a component cross-swap only if leave-one-fold-out selection improves >=0.003",
        "  in >=4/5 held-out folds, then confirm the fixed pair on strict outer-location OOF.",
        "- Keep tile/global champions in separate registries until the server aggregation is known.",
        "",
        "Outputs: `factorization_summary.csv`, `tile_factorization.csv`, `cross_swap.csv`,",
        "`cross_swap_lofo.csv`,",
        "`metric_scale_stress.csv`, `tail_concentration.csv`, `strict_exp011_additive.csv`,",
        "`availability_summary.csv`, `location_bootstrap.csv`, `summary.json`, `run_manifest.json`.",
    ]
    return "\n".join(lines) + "\n"


def run_self_test() -> None:
    rng = np.random.default_rng(123)
    target = rng.gamma(1.3, 0.8, size=(17, 4, 4)).astype(np.float32)
    target[0] = 0.0
    pred = np.maximum(0.0, 0.83 * target + rng.normal(0.0, 0.25, target.shape)).astype(np.float32)
    yy = np.einsum("nij,nij->n", target, target, dtype=np.float64) / 16
    ymean = target.mean(axis=(1, 2), dtype=np.float64)
    moments = field_moments(pred, target, yy)
    record = exact_factorization(moments, ymean, yy)
    direct = np.square(pred - target).mean(axis=(1, 2), dtype=np.float64)
    if not np.allclose(record["total_mse"], direct, atol=1e-6):
        raise AssertionError("quadratic field MSE does not match direct MSE")
    mean_scaled = np.empty_like(pred)
    for i in range(len(pred)):
        if record["mean"][i] > EPS:
            mean_scaled[i] = pred[i] * record["mean_scale"][i]
        else:
            mean_scaled[i] = ymean[i]
    direct_mean_scaled = np.square(mean_scaled - target).mean(axis=(1, 2), dtype=np.float64)
    if not np.allclose(record["mean_scale_mse"], direct_mean_scaled, atol=1e-6):
        raise AssertionError("mean-scale quadratic formula does not match direct evaluation")
    diagonal = cross_swap_mse(moments, moments, ymean, yy)
    if not np.allclose(diagonal, record["total_mse"], atol=1e-6):
        raise AssertionError("amount x own-shape diagonal does not reconstruct the field")
    scale = 1.17
    quadratic = safe_mse(scale * scale * moments["pp"] - 2 * scale * moments["py"] + yy)
    direct_scaled = np.square(scale * pred - target).mean(axis=(1, 2), dtype=np.float64)
    if not np.allclose(quadratic, direct_scaled, atol=1e-6):
        raise AssertionError("scale stress quadratic formula does not match direct evaluation")
    print("self-test passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-tiles", type=int, default=None, help="smoke-test prefix after ID sort")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260716)
    parser.add_argument("--scale-min", type=float, default=0.5)
    parser.add_argument("--scale-max", type=float, default=1.5)
    parser.add_argument("--scale-step", type=float, default=0.025)
    parser.add_argument("--skip-tile-csv", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    if args.max_tiles is not None and args.max_tiles <= 0:
        raise ValueError("--max-tiles must be positive")
    if args.bootstrap_reps <= 0:
        raise ValueError("--bootstrap-reps must be positive")

    started = time.time()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"loading caches from {CACHE_DIR}", flush=True)
    bundle = load_bundle(args.max_tiles)
    target = bundle["target"]
    ymean = target.mean(axis=(1, 2), dtype=np.float64)
    yy = np.einsum("nij,nij->n", target, target, dtype=np.float64, optimize=True) / PIXELS
    wet_fraction = (target > 0).mean(axis=(1, 2), dtype=np.float64)
    groups = build_groups(bundle, ymean, wet_fraction)

    moments: dict[str, dict[str, np.ndarray]] = {}
    diagnostics: dict[str, dict[str, np.ndarray]] = {}
    factor_rows: list[dict[str, object]] = []
    for experiment in EXPERIMENTS:
        print(f"factorizing {experiment}", flush=True)
        moments[experiment] = field_moments(bundle["predictions"][experiment], target, yy)
        diagnostics[experiment] = exact_factorization(moments[experiment], ymean, yy)
        factor_rows += factorization_summary_rows(experiment, diagnostics[experiment], groups)
    write_csv(out_dir / "factorization_summary.csv", factor_rows)
    if not args.skip_tile_csv:
        print("writing tile-level diagnostics", flush=True)
        write_tile_diagnostics(out_dir / "tile_factorization.csv", bundle, ymean, wet_fraction, diagnostics)

    print("running strict exp011 additive reference", flush=True)
    strict_rows, strict_arrays = strict_additive_reference(
        bundle["unique_id"], groups, require_exact_ids=args.max_tiles is None
    )
    write_csv(out_dir / "strict_exp011_additive.csv", strict_rows)

    print("running amount x shape cross-swap", flush=True)
    swap_rows, swap_summary, _, swap_lofo_rows = run_cross_swap(
        moments, ymean, yy, groups
    )
    write_csv(out_dir / "cross_swap.csv", swap_rows)
    write_csv(out_dir / "cross_swap_lofo.csv", swap_lofo_rows)

    print("running tile/global metric stress", flush=True)
    stress_rows, stress_summary, _, tail_rows = run_metric_stress(
        bundle["predictions"], target, yy, groups,
        args.scale_min, args.scale_max, args.scale_step,
    )
    write_csv(out_dir / "metric_scale_stress.csv", stress_rows)
    write_csv(out_dir / "tail_concentration.csv", tail_rows)

    availability = availability_rows(
        bundle, ymean, wet_fraction, strict_arrays, diagnostics["exp018"]
    )
    write_csv(out_dir / "availability_summary.csv", availability)

    print(f"running {args.bootstrap_reps} location-cluster bootstrap replicates", flush=True)
    boot_rows = bootstrap_rows(
        diagnostics, strict_arrays, bundle["location"], bundle["satellite"],
        args.bootstrap_reps, args.bootstrap_seed
    )
    write_csv(out_dir / "location_bootstrap.csv", boot_rows)

    summary = {
        "n_tiles": len(bundle["unique_id"]),
        "experiments": list(EXPERIMENTS),
        "risk": {
            "exp016_exp017_exp018": "amber_successor_screening",
            "exp011": "strict_row_only_reference",
            "target_oracles": "diagnostic_only_never_submission",
        },
        "factorization_global": {
            exp: global_row(factor_rows, exp) for exp in EXPERIMENTS
        },
        "strict_exp011_global": global_row(strict_rows, "exp011"),
        "cross_swap": swap_summary,
        "metric_stress": stress_summary,
        "own_row_observation_counts": {
            str(row["own_row_observation_count"]): row["samples"]
            for row in availability if row["satellite"] == "all"
        },
        "bootstrap": boot_rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default), encoding="utf-8"
    )
    source_paths = list(bundle["cache_paths"]) + [TRAIN_CSV, STRICT_SAMPLE_CSV]
    if WEIGHT_JSON.exists():
        source_paths.append(WEIGHT_JSON)
    source_records = [file_record(path) for path in source_paths]
    elapsed = time.time() - started
    report = render_report(
        factor_rows, strict_rows, swap_summary, stress_summary,
        availability, boot_rows, elapsed, args.max_tiles,
    )
    (out_dir / "EDA_REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": script_sha256(),
        "created_unix": time.time(),
        "elapsed_seconds": elapsed,
        "arguments": vars(args),
        "data_contract": bundle["data_contract"],
        "weight_source": stress_summary["weight_source"],
        "optional_weight_file_present": WEIGHT_JSON.exists(),
        "sources": source_records,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=json_default), encoding="utf-8"
    )
    print(report, flush=True)
    print(f"wrote outputs to {out_dir} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
