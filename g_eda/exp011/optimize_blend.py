#!/usr/bin/env python3
"""g_eda/exp011: manifest-driven OOF blend-weight optimizer for the green-only Track G3 blend.

Direct successor to g_eda/exp003's OOF simplex search, generalized so that adding a new green
source is a one-entry edit to sources.json instead of a new script (exp003 grew a new
run_Nsource_blend.py file for every additional source -- see run_4source_blend.py /
run_5source_blend.py / run_4way_simplex.py). All of exp003's caches/recommendations are
red-derived (exp016/017/018-family checkpoints) per doc/submission_registry.md and are NOT read
here; this experiment starts a clean, green-only OOF cache tree under
outputs/g_eda/exp011/.

Phase 1 (GPU or CPU, once per source): regenerate OOF predictions fold-by-fold from the
checkpoints named in sources.json and cache them as fp16 npz (reusable, same schema exp003 used:
pred/target/unique_id/satellite/fold arrays).

    python3 optimize_blend.py --cache exp038_sigmafixed
    python3 optimize_blend.py --cache exp040_metric

(`--cache NAME` reads module_dir/checkpoint_dir for NAME straight out of sources.json -- no
need to repeat them on the command line the way g_eda/exp003's entrypoint required.)

Phase 2 (CPU only): compute the OOF-optimal blend from every source currently listed in
sources.json.

    python3 optimize_blend.py --analyze

Every source is passed through `registry_guard.assert_green` before its cache is touched --
this script raises rather than blending in anything not currently green (see registry_guard.py).

Search strategy is chosen automatically by source count N (see `search()`):
  - N == 1: trivial, weight 1.0 (nothing to optimize).
  - N == 2: full 1-D ladder w in [0, 1] step 0.01 (finer than exp003's 0.05 since a 1-D sweep is
    cheap), global and per-satellite.
  - N == 3: full simplex grid step 0.05 (matches exp003's 3-way exactly).
  - N > 3: greedy forward blend-in -- start from the best solo source, then repeatedly fold in
    whichever remaining source most improves OOF tile_rmse at its own optimal single blend-in
    weight (mirrors exp003's run_4source_blend.py / run_5source_blend.py approximation, which
    that experiment measured to be within 0.00008 of the true simplex optimum). A full N-simplex
    grid at fine resolution becomes intractable past ~4 sources; this keeps the search tractable
    without a code rewrite as exp047/050/051/052/053/054 land.

Outputs: recommended_weights.json + BLEND_CURVE.md (mirrors exp003's report), plus
blend_curve.csv / simplex_grid.csv / greedy_path.csv depending on which branch ran.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent
OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp011"
SOURCES_JSON = EXP_DIR / "sources.json"
CAUSAL_JSON = PROJECT_DIR / "g_eda" / "exp010" / "recommended_causal_weights.json"
SATELLITES = ("goes", "himawari", "meteosat")

sys.path.insert(0, str(EXP_DIR))
import registry_guard  # noqa: E402


def load_manifest() -> list[dict]:
    manifest = json.loads(SOURCES_JSON.read_text(encoding="utf-8"))
    return manifest["sources"]


def source_by_name(name: str, manifest: list[dict] | None = None) -> dict:
    manifest = manifest or load_manifest()
    for entry in manifest:
        if entry["name"] == name:
            return entry
    raise KeyError(f"'{name}' not found in {SOURCES_JSON} -- add it to sources.json first")


# ---------------------------------------------------------------- phase 1: cache OOF preds

def cache_path(source_name: str) -> Path:
    return OUT_DIR / f"{source_name}_oof_pred.npz"


def build_cache(source_name: str, batch_size: int, num_workers: int, device_arg: str) -> None:
    registry_guard.assert_green(source_name)
    entry = source_by_name(source_name)

    import torch
    from torch.utils.data import DataLoader

    exp_dir = PROJECT_DIR / "g_experiments" / entry["module_dir"]
    sys.path.insert(0, str(exp_dir))
    import dataset as dataset_mod  # noqa: E402
    import model as model_mod  # noqa: E402

    if device_arg == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_arg)
    print(f"{source_name}: using device={device}", flush=True)

    ckpt_dir = PROJECT_DIR / "g_model" / entry["checkpoint_dir"]
    checkpoints = sorted(ckpt_dir.glob("best_model_fold*.pt"))
    if len(checkpoints) != 5:
        raise FileNotFoundError(f"{source_name}: expected 5 checkpoints, found {len(checkpoints)} in {ckpt_dir}")

    preds, targets, unique_ids, satellites, folds = [], [], [], [], []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        fold = int(checkpoint["fold"])
        model = model_mod.build_model(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        rows = dataset_mod.read_rows((exp_dir / config["data"]["train_csv"]).resolve())
        _, valid_rows, _ = dataset_mod.make_group_kfold_split(
            rows, n_splits=int(config["split"]["n_splits"]), fold=fold, seed=int(config["experiment"]["seed"])
        )
        ds_kwargs = {}
        if hasattr(dataset_mod, "features_from_config"):
            ds_kwargs["features"] = dataset_mod.features_from_config(config)
        ds = dataset_mod.PrecipDataset(
            valid_rows,
            (exp_dir / config["data"]["train_dir"]).resolve(),
            max_observations=int(config["data"]["max_observations"]),
            satellite_channels=int(config["data"]["satellite_channels"]),
            target_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
            context_rows=int(config["data"].get("context_rows", 1)),
            has_target=True,
            norm_stats=dataset_mod.load_norm_stats((exp_dir / config["paths"]["norm_stats"]).resolve()),
            augment=False,
            **ds_kwargs,
        )
        if int(config["data"].get("context_rows", 1)) != 1:
            raise RegistryGuardMismatch(  # noqa: F821 - defined below, keeps this file single-pass readable
                f"{source_name} fold={fold}: context_rows={config['data'].get('context_rows')} != 1 "
                "-- this is a successor-row (red) config and must never enter the green cache."
            )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"))
        clip_min = float(config["model"]["clip_min"])
        print(f"{source_name} fold={fold} rows={len(ds)}", flush=True)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].float()
                pred = model_mod.prediction_from_output(model(x)).float().clamp_min(clip_min)
                preds.append(pred.squeeze(1).cpu().numpy().astype(np.float16))
                targets.append(y.squeeze(1).numpy().astype(np.float16))
                unique_ids.extend(batch["unique_id"])
                satellites.extend(batch["satellite_target"])
                folds.extend([fold] * len(batch["unique_id"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(source_name),
        pred=np.concatenate(preds),
        target=np.concatenate(targets),
        unique_id=np.asarray(unique_ids),
        satellite=np.asarray(satellites),
        fold=np.asarray(folds, dtype=np.int8),
    )
    print(f"cached {cache_path(source_name)}", flush=True)


class RegistryGuardMismatch(RuntimeError):
    """Raised when a manifest entry's actual training config isn't context_rows: 1 (green)."""


# ---------------------------------------------------------------- phase 2: analysis

def tile_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(pred - target).reshape(pred.shape[0], -1).mean(axis=1))


def load_aligned(manifest: list[dict]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, int]:
    names = [entry["name"] for entry in manifest]
    registry_guard.assert_all_green(names)

    caches = {}
    for entry in manifest:
        path = OUT_DIR / f"{entry['name']}_oof_pred.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing -- run `python3 optimize_blend.py --cache {entry['name']}` first"
            )
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
    n = target.shape[0]
    return aligned, target, satellite, n


def make_score_fn(target: np.ndarray, satellite: np.ndarray):
    sat_masks = {sat: satellite == sat for sat in SATELLITES}

    def score(pred: np.ndarray) -> dict[str, float]:
        per_tile = tile_rmse(pred, target)
        result = {"overall": float(per_tile.mean())}
        for sat, mask in sat_masks.items():
            if mask.any():
                result[sat] = float(per_tile[mask].mean())
        return result

    return score, sat_masks


def search_two_way(names: list[str], aligned: dict[str, np.ndarray], target: np.ndarray,
                    sat_masks: dict[str, np.ndarray], score) -> dict:
    a, b = names
    curve_rows = []
    for w in np.round(np.arange(0.0, 1.0001, 0.01), 2):
        pred = (1.0 - w) * aligned[a] + w * aligned[b]
        curve_rows.append({f"w_{b}": float(w), **score(pred)})
    best_global = min(curve_rows, key=lambda r: r["overall"])

    per_sat_best = {}
    for sat, mask in sat_masks.items():
        if not mask.any():
            continue
        best = None
        for w in np.round(np.arange(0.0, 1.0001, 0.01), 2):
            pred = (1.0 - w) * aligned[a][mask] + w * aligned[b][mask]
            value = float(tile_rmse(pred, target[mask]).mean())
            if best is None or value < best[1]:
                best = (w, value)
        per_sat_best[sat] = {"weights": {a: round(1.0 - best[0], 4), b: round(best[0], 4)},
                             "overall": best[1]}

    composed = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        if sat not in per_sat_best:
            continue
        w = per_sat_best[sat]["weights"]
        composed[mask] = w[a] * aligned[a][mask] + w[b] * aligned[b][mask]
    composed_score = score(composed)

    with (OUT_DIR / "blend_curve.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    return {
        "method": "2-way ladder (step 0.01)",
        "global_best": {"weights": {a: round(1.0 - best_global[f"w_{b}"], 4),
                                     b: round(best_global[f"w_{b}"], 4)},
                        **{k: v for k, v in best_global.items() if k != f"w_{b}"}},
        "per_satellite_best": per_sat_best,
        "per_satellite_composed": composed_score,
        "curve_csv": "blend_curve.csv",
    }


def search_three_way(names: list[str], aligned: dict[str, np.ndarray], target: np.ndarray,
                      sat_masks: dict[str, np.ndarray], score) -> dict:
    step = 0.05
    steps = int(round(1.0 / step))
    grid_rows = []
    for i, j in itertools.product(range(steps + 1), repeat=2):
        if i + j > steps:
            continue
        w0, w1 = i * step, j * step
        w2 = 1.0 - w0 - w1
        pred = w0 * aligned[names[0]] + w1 * aligned[names[1]] + w2 * aligned[names[2]]
        grid_rows.append({names[0]: round(w0, 2), names[1]: round(w1, 2), names[2]: round(w2, 2),
                          **score(pred)})
    best_global = min(grid_rows, key=lambda r: r["overall"])

    per_sat_best = {}
    for sat, mask in sat_masks.items():
        if not mask.any():
            continue
        best = None
        for row in grid_rows:
            weights = {n: row[n] for n in names}
            pred = sum(weights[n] * aligned[n][mask] for n in names)
            value = float(tile_rmse(pred, target[mask]).mean())
            if best is None or value < best[1]:
                best = (weights, value)
        per_sat_best[sat] = {"weights": best[0], "overall": best[1]}

    composed = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        if sat not in per_sat_best:
            continue
        w = per_sat_best[sat]["weights"]
        composed[mask] = sum(w[n] * aligned[n][mask] for n in names)
    composed_score = score(composed)

    with (OUT_DIR / "simplex_grid.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(grid_rows)

    return {
        "method": "3-way simplex grid (step 0.05)",
        "global_best": {"weights": {n: best_global[n] for n in names},
                        **{k: v for k, v in best_global.items() if k not in names}},
        "per_satellite_best": per_sat_best,
        "per_satellite_composed": composed_score,
        "grid_csv": "simplex_grid.csv",
    }


def search_greedy(names: list[str], aligned: dict[str, np.ndarray], target: np.ndarray,
                   sat_masks: dict[str, np.ndarray], score) -> dict:
    """N > 3: greedy forward blend-in, matching exp003's run_4source_blend.py /
    run_5source_blend.py pattern (fix the current best blend, sweep one new source's blend-in
    weight per satellite, repeat). Order is chosen greedily by whichever remaining source
    improves OOF the most at each step.
    """
    solo_scores = {n: score(aligned[n])["overall"] for n in names}
    remaining = set(names)
    first = min(remaining, key=lambda n: solo_scores[n])
    remaining.remove(first)

    current_weights = {sat: {first: 1.0} for sat in sat_masks if sat_masks[sat].any()}
    current_pred = {sat: aligned[first][mask].copy() for sat, mask in sat_masks.items() if mask.any()}
    path_rows = [{"step": 0, "added": first, "overall": solo_scores[first]}]

    while remaining:
        best_choice = None  # (name, per_sat_w, per_sat_pred, overall)
        for candidate in remaining:
            per_sat_w = {}
            per_sat_pred = {}
            for sat, mask in sat_masks.items():
                if not mask.any():
                    continue
                best = None
                for w_new in np.round(np.arange(0.0, 1.0001, 0.05), 2):
                    pred = (1.0 - w_new) * current_pred[sat] + w_new * aligned[candidate][mask]
                    value = float(tile_rmse(pred, target[mask]).mean())
                    if best is None or value < best[1]:
                        best = (w_new, value, pred)
                per_sat_w[sat] = best[0]
                per_sat_pred[sat] = best[2]
            combined = np.zeros_like(target)
            for sat, mask in sat_masks.items():
                if sat in per_sat_pred:
                    combined[mask] = per_sat_pred[sat]
            overall = score(combined)["overall"]
            if best_choice is None or overall < best_choice[3]:
                best_choice = (candidate, per_sat_w, per_sat_pred, overall)

        chosen, per_sat_w, per_sat_pred, overall = best_choice
        for sat in current_weights:
            w_new = per_sat_w[sat]
            for name in current_weights[sat]:
                current_weights[sat][name] *= (1.0 - w_new)
            current_weights[sat][chosen] = current_weights[sat].get(chosen, 0.0) + w_new
            current_pred[sat] = per_sat_pred[sat]
        remaining.remove(chosen)
        path_rows.append({"step": len(path_rows), "added": chosen, "overall": overall})

    combined = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        if sat in current_pred:
            combined[mask] = current_pred[sat]
    composed_score = score(combined)

    # global_best: same greedy path but with a single shared weight set (no per-satellite split)
    remaining = set(names)
    g_first = min(remaining, key=lambda n: solo_scores[n])
    remaining.remove(g_first)
    g_weights = {g_first: 1.0}
    g_pred = aligned[g_first].copy()
    g_path = [{"step": 0, "added": g_first, "overall": solo_scores[g_first]}]
    while remaining:
        best = None
        for candidate in remaining:
            for w_new in np.round(np.arange(0.0, 1.0001, 0.05), 2):
                pred = (1.0 - w_new) * g_pred + w_new * aligned[candidate]
                value = float(tile_rmse(pred, target).mean())
                if best is None or value < best[1]:
                    best = (candidate, w_new, value, pred)
        chosen, w_new, value, pred = best
        for name in g_weights:
            g_weights[name] *= (1.0 - w_new)
        g_weights[chosen] = g_weights.get(chosen, 0.0) + w_new
        g_pred = pred
        remaining.remove(chosen)
        g_path.append({"step": len(g_path), "added": chosen, "overall": value})

    with (OUT_DIR / "greedy_path.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "added", "overall"])
        writer.writeheader()
        writer.writerows(path_rows)

    return {
        "method": f"greedy forward blend-in (N={len(names)} > 3, step 0.05 per addition)",
        "global_best": {"weights": {n: round(w, 4) for n, w in g_weights.items()},
                        "overall": g_path[-1]["overall"]},
        "per_satellite_best": {sat: {"weights": {n: round(w, 4) for n, w in ws.items()}}
                               for sat, ws in current_weights.items()},
        "per_satellite_composed": composed_score,
        "greedy_path_csv": "greedy_path.csv",
    }


def search(manifest: list[dict]) -> dict:
    names = [entry["name"] for entry in manifest]
    aligned, target, satellite, n = load_aligned(manifest)
    score, sat_masks = make_score_fn(target, satellite)
    solo = {name: score(aligned[name]) for name in names}

    if len(names) == 1:
        result = {
            "method": "single source (nothing to optimize)",
            "global_best": {"weights": {names[0]: 1.0}, **solo[names[0]]},
            "per_satellite_best": {sat: {"weights": {names[0]: 1.0}} for sat in sat_masks if sat_masks[sat].any()},
            "per_satellite_composed": solo[names[0]],
        }
    elif len(names) == 2:
        result = search_two_way(names, aligned, target, sat_masks, score)
    elif len(names) == 3:
        result = search_three_way(names, aligned, target, sat_masks, score)
    else:
        result = search_greedy(names, aligned, target, sat_masks, score)

    causal_hook = {"available": CAUSAL_JSON.exists(), "path": str(CAUSAL_JSON)}
    if causal_hook["available"]:
        causal_rec = json.loads(CAUSAL_JSON.read_text(encoding="utf-8"))
        smoothing = causal_rec.get("temporal_smoothing", {})
        if smoothing.get("next_weight", 0.0) not in (0, 0.0):
            raise RegistryGuardMismatch(
                "g_eda/exp010's recommendation has next_weight != 0 -- refusing to wire in a "
                "non-causal smoothing recommendation (2026-07-20 ruling)."
            )
        causal_hook["temporal_smoothing"] = smoothing
        causal_hook["source_experiment"] = causal_rec.get("source_experiment")
    else:
        causal_hook["note"] = ("g_eda/exp010/recommended_causal_weights.json does not exist yet "
                               "-- causal-only smoothing hook is wired but inactive until that "
                               "experiment lands its recommendation.")

    recommendation = {
        "source": "g_eda/exp011 manifest-driven OOF blend optimization",
        "manifest_sources": names,
        "n_tiles": n,
        "solo_scores": solo,
        **result,
        "causal_smoothing_hook": causal_hook,
    }
    return recommendation


def write_report(rec: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "recommended_weights.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    lines = ["# OOF blend optimization (g_eda/exp011)", "",
             f"Sources in this run: {', '.join(rec['manifest_sources'])}",
             f"Method: {rec['method']}",
             f"Tiles: {rec['n_tiles']}", ""]
    lines.append("## Solo OOF scores")
    lines.append("")
    for name, s in rec["solo_scores"].items():
        lines.append(f"- {name}: {s['overall']:.5f}"
                      + " (" + ", ".join(f"{sat}={s[sat]:.5f}" for sat in SATELLITES if sat in s) + ")")
    lines.append("")
    gb = rec["global_best"]
    lines.append(f"## Global best blend: {json.dumps(gb['weights'])} -> {gb['overall']:.5f}")
    best_solo_name = min(rec["solo_scores"], key=lambda n: rec["solo_scores"][n]["overall"])
    best_solo = rec["solo_scores"][best_solo_name]["overall"]
    delta = gb["overall"] - best_solo
    lines.append(f"- delta vs best solo ({best_solo_name} {best_solo:.5f}): {delta:+.5f}")
    lines.append("")
    lines.append("## Per-satellite composed blend")
    lines.append("")
    psc = rec["per_satellite_composed"]
    lines.append(f"- overall: {psc['overall']:.5f} ("
                  + ", ".join(f"{sat}={psc[sat]:.5f}" for sat in SATELLITES if sat in psc) + ")")
    for sat, info in rec["per_satellite_best"].items():
        lines.append(f"  - {sat}: {json.dumps(info['weights'])}")
    lines.append("")
    lines.append("## Causal-only smoothing hook (g_eda/exp010)")
    lines.append("")
    hook = rec["causal_smoothing_hook"]
    if hook["available"]:
        lines.append(f"- LOADED from {hook['path']} (source_experiment={hook.get('source_experiment')})")
        lines.append(f"  - {json.dumps(hook['temporal_smoothing'])}")
    else:
        lines.append(f"- PENDING: {hook['note']}")
    lines.append("")
    lines.append("Full grids in blend_curve.csv / simplex_grid.csv / greedy_path.csv (whichever "
                 "applies to this source count); g_experiments/exp055 consumes recommended_weights.json.")
    (OUT_DIR / "BLEND_CURVE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", metavar="NAME", help="phase 1: cache OOF preds for one manifest source")
    parser.add_argument("--analyze", action="store_true", help="phase 2: compute blend weights from caches")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    started = time.time()
    if args.cache:
        build_cache(args.cache, args.batch_size, args.num_workers, args.device)
    if args.analyze:
        manifest = load_manifest()
        rec = search(manifest)
        write_report(rec)
    if not args.cache and not args.analyze:
        raise SystemExit("specify --cache NAME and/or --analyze")
    print(f"done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
