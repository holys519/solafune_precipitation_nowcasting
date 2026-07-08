#!/usr/bin/env python3
"""Summarize available ensemble sources for exp007."""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    args = parser.parse_args()

    config = load_config(Path(args.config))
    analysis_dir = resolve_path(config["paths"].get("analysis_dir", "../../outputs/analysis/exp007"))
    analysis_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for source in config.get("ensemble", {}).get("sources", []):
        name = str(source["name"])
        model_dir = resolve_path(str(source["model_dir"]))
        checkpoints = sorted(model_dir.glob("best_model_fold*.pt")) if source.get("enabled", True) else []
        metric_paths = sorted(model_dir.glob("metrics_fold*.json")) if source.get("enabled", True) else []
        best_values = []
        for path in metric_paths:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                best_values.append(float(data.get("best_rmse", "nan")))
            except Exception as exc:  # noqa: BLE001 - diagnostic script should keep going.
                print(f"WARNING: failed reading {path}: {exc}", flush=True)
        rows.append(
            {
                "source": name,
                "enabled": bool(source.get("enabled", True)),
                "model_dir": str(model_dir),
                "configured_weight": float(source.get("weight", 1.0)),
                "checkpoints": len(checkpoints),
                "metrics": len(metric_paths),
                "best_rmse_mean": sum(best_values) / len(best_values) if best_values else "",
                "best_rmse_min": min(best_values) if best_values else "",
                "best_rmse_max": max(best_values) if best_values else "",
            }
        )

    source_path = analysis_dir / "ensemble_source_summary.csv"
    with source_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "source",
            "enabled",
            "model_dir",
            "configured_weight",
            "checkpoints",
            "metrics",
            "best_rmse_mean",
            "best_rmse_min",
            "best_rmse_max",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {"sources": rows, "source_summary_csv": str(source_path)}
    (analysis_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
