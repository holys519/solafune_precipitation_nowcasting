#!/usr/bin/env python3
"""One-off recovery script (2026-07-30): exp064_convnext_small_lr2e4 fold3's metrics_fold3.json
was written as a 0-byte file because the disk-full incident hit exactly during train.py's final
`metrics_path.write_text(...)` call (outside the training loop, so it never got a chance to
retry). The checkpoint (`best_model_fold3.pt`) and the per-epoch `training_log_fold3.csv` (written
incrementally every epoch, unaffected) are both intact -- this reconstructs a metrics_fold3.json
byte-for-byte schema-compatible with what train.py itself would have written, from those two
sources, so downstream tools (analyze_oof.py's summarize_training, which crashed on the empty
file) work again without needing to retrain fold3 from scratch.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml

from dataset import make_group_kfold_split, read_rows

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config_convnext_small_lr2e4.yaml"
CSV_PATH = SCRIPT_DIR.parent.parent / "outputs" / "analysis" / "exp064_convnext_small_lr2e4" / "training_log_fold3.csv"
OUT_PATH = SCRIPT_DIR.parent.parent / "g_model" / "exp064_convnext_small_lr2e4" / "metrics_fold3.json"
FOLD = 3


def main() -> None:
    if OUT_PATH.stat().st_size != 0:
        raise SystemExit(f"{OUT_PATH} is not empty ({OUT_PATH.stat().st_size} bytes) -- refusing "
                          "to overwrite a file that isn't the known-corrupt 0-byte one.")

    config = yaml.safe_load(CONFIG_PATH.read_text())
    seed = int(config["experiment"]["seed"])
    n_splits = int(config["split"]["n_splits"])
    train_csv = (SCRIPT_DIR / config["data"]["train_csv"]).resolve()
    rows = read_rows(train_csv)
    _, _, valid_locations = make_group_kfold_split(rows, n_splits=n_splits, fold=FOLD, seed=seed)

    with CSV_PATH.open(newline="") as f:
        csv_rows = list(csv.DictReader(f))
    if not csv_rows:
        raise SystemExit(f"{CSV_PATH} has no rows -- cannot reconstruct")

    history = []
    for row in csv_rows:
        history.append({
            "epoch": int(row["epoch"]),
            "train_rmse": float(row["train_rmse"]),
            "train_tile_rmse": float(row["train_tile_rmse"]),
            "rmse": float(row["valid_rmse"]),
            "tile_rmse": float(row["valid_tile_rmse"]),
            "zero_rmse": float(row["zero_rmse"]),
            "positive_rmse": float(row["positive_rmse"]),
            "pixels": float(row["pixels"]),
            "positive_pixels": float(row["positive_pixels"]),
            "samples": float(row["samples"]),
            "elapsed_seconds": float(row["elapsed_seconds"]),
            "lr": float(row["lr"]),
        })

    selection_metric = csv_rows[0]["selection_metric"]
    best_row = min(history, key=lambda h: h[selection_metric if selection_metric in h else "rmse"])
    best_epoch = best_row["epoch"]
    best_rmse = best_row["rmse"]
    best_tile_rmse = best_row["tile_rmse"]
    best_metric = best_row[selection_metric] if selection_metric in best_row else best_row["rmse"]

    # early_stopping_patience=20 (this recovery config); stopped_early is true iff the run ended
    # before config['train']['epochs'] (100) -- 32 completed epochs confirms it triggered.
    epochs_configured = int(config["train"]["epochs"])
    stopped_early = len(history) < epochs_configured

    payload = {
        "fold": FOLD,
        "valid_locations": valid_locations,
        "history": history,
        "best_rmse": best_rmse,
        "best_tile_rmse": best_tile_rmse,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "initial_metrics": None,
        "initialized_from": None,
        "stopped_early": stopped_early,
        "epochs_completed": len(history),
        "train_rows_used": None,
        "valid_rows_used": None,
        "torch_version": None,
        "cuda_devices": [],
        "training_log": str(CSV_PATH),
        "_reconstructed": True,
        "_reconstruction_note": (
            "2026-07-30: original metrics_fold3.json was a 0-byte file from a disk-full write "
            "failure at the very end of training. Rebuilt from training_log_fold3.csv (unaffected, "
            "written incrementally every epoch) plus a fresh, deterministic re-derivation of "
            "valid_locations via make_group_kfold_split(fold=3, seed=42) -- identical to what "
            "train.py computed originally. train_rows_used/valid_rows_used/torch_version/"
            "cuda_devices/initial_metrics/initialized_from could not be recovered (never logged "
            "anywhere but the lost final write) and are set to null/empty; nothing downstream "
            "(analyze_oof.py's summarize_training) reads those fields."
        ),
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT_PATH} (best_epoch={best_epoch}, best_tile_rmse={best_tile_rmse:.5f}, "
          f"epochs_completed={len(history)}, stopped_early={stopped_early}, "
          f"valid_locations={valid_locations})")


if __name__ == "__main__":
    main()
