#!/usr/bin/env python3
"""Summarize exp041 fold-0/fold-4 screen without reading prediction fields."""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "outputs" / "analysis" / "exp041_screen"
ARMS = {
    "control": PROJECT_DIR / "g_model" / "exp041_control",
    "metric": PROJECT_DIR / "g_model" / "exp041_metric",
}
BASE = PROJECT_DIR / "g_model" / "exp038"
FOLDS = (0, 4)


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for fold in FOLDS:
        base = load(BASE / f"metrics_fold{fold}.json")
        if base is None:
            raise FileNotFoundError(BASE / f"metrics_fold{fold}.json")
        for arm, model_dir in ARMS.items():
            metrics_path = model_dir / f"metrics_fold{fold}.json"
            metrics = load(metrics_path)
            rows.append(
                {
                    "arm": arm,
                    "fold": fold,
                    "status": "complete" if metrics is not None else "missing",
                    "samples": int(base.get("valid_rows_used", 0)),
                    "base_exp038": float(base["best_tile_rmse"]),
                    "initial_tile_rmse": (
                        float(metrics["initial_metrics"]["tile_rmse"])
                        if metrics is not None and metrics.get("initial_metrics")
                        else ""
                    ),
                    "best_tile_rmse": float(metrics["best_tile_rmse"]) if metrics is not None else "",
                    "delta_vs_base": (
                        float(metrics["best_tile_rmse"]) - float(base["best_tile_rmse"])
                        if metrics is not None
                        else ""
                    ),
                    "best_epoch": metrics.get("best_epoch", "") if metrics is not None else "",
                    "epochs_completed": metrics.get("epochs_completed", "") if metrics is not None else "",
                    "metrics_path": str(metrics_path.relative_to(PROJECT_DIR)),
                }
            )

    by_key = {(str(row["arm"]), int(row["fold"])): row for row in rows}
    complete = all(row["status"] == "complete" for row in rows)
    metric_better_both = False
    weighted_delta = None
    initial_ok = False
    if complete:
        deltas = []
        weighted_sum = 0.0
        sample_sum = 0
        initial_ok = True
        for fold in FOLDS:
            control = by_key[("control", fold)]
            metric = by_key[("metric", fold)]
            delta = float(metric["best_tile_rmse"]) - float(control["best_tile_rmse"])
            deltas.append(delta)
            samples = int(metric["samples"])
            weighted_sum += samples * delta
            sample_sum += samples
            for row in (control, metric):
                initial_ok &= abs(float(row["initial_tile_rmse"]) - float(row["base_exp038"])) < 1e-6
        metric_better_both = all(delta < 0 for delta in deltas)
        weighted_delta = weighted_sum / sample_sum

    gate_pass = bool(
        complete
        and initial_ok
        and metric_better_both
        and weighted_delta is not None
        and weighted_delta <= -0.002
        and all(float(by_key[("metric", fold)]["delta_vs_base"]) < 0 for fold in FOLDS)
    )
    fieldnames = list(rows[0])
    with (OUT_DIR / "screen_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "complete": complete,
        "initial_checkpoint_reproduction_ok": initial_ok,
        "metric_better_both_folds": metric_better_both,
        "sample_weighted_metric_minus_control": weighted_delta,
        "gate_pass": gate_pass,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        "# exp041 screening report",
        "",
        "| Arm | Fold | Status | exp038 | Initial | Best | Δ vs exp038 | Best epoch |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def value(key: str) -> str:
            item = row[key]
            return f"{float(item):.6f}" if item != "" and key not in {"best_epoch"} else str(item)
        lines.append(
            f"| {row['arm']} | {row['fold']} | {row['status']} | {value('base_exp038')} "
            f"| {value('initial_tile_rmse')} | {value('best_tile_rmse')} "
            f"| {value('delta_vs_base')} | {row['best_epoch']} |"
        )
    lines += [
        "",
        f"- Complete: `{complete}`",
        f"- Initial checkpoint reproduction: `{initial_ok}`",
        f"- Metric better on both folds: `{metric_better_both}`",
        f"- Sample-weighted metric−control: `{weighted_delta}`",
        f"- Gate pass: `{gate_pass}`",
        "",
    ]
    (OUT_DIR / "SCREENING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
