#!/usr/bin/env python3
"""exp027: seed-ensemble blend + tile-overlap patch.

1. Run exp017 inference for each exp025 seed checkpoint set (5-fold ensemble each).
2. Blend exp016 + exp017 + exp017-seed{42,123,2026} evaluation predictions.
3. Apply the exp014 tile-overlap GPM copy patch to every blend scheme.
4. Emit patched submission zips: outputs/submissions/exp027_<scheme>_patched.zip
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXP014 = ROOT / "g_experiments/exp014"
EXP017 = ROOT / "g_experiments/exp017"
SUB = ROOT / "outputs/submissions"
ANA = ROOT / "outputs/analysis/exp027"

sys.path.insert(0, str(EXP017))
from tiff_utils import read_tiff_array, write_float32_like_template  # noqa: E402


def run_seed_inference(seed: int, force: bool) -> Path:
    out_dir = SUB / f"exp027/seed{seed}"
    test_files = out_dir / "test_files"
    if not force and test_files.is_dir() and any(test_files.glob("*.tif")):
        print(f"seed{seed}: predictions already exist, skipping inference", flush=True)
        return out_dir

    checkpoints = sorted((ROOT / f"g_model/exp025/seed{seed}").glob("best_model_fold*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under g_model/exp025/seed{seed} (run exp025 first)")
    if len(checkpoints) < 5:
        print(f"WARNING: seed{seed} has only {len(checkpoints)} fold checkpoints", flush=True)

    src_cfg = ROOT / f"outputs/analysis/exp025/seed{seed}/config.yaml"
    config = yaml.safe_load(src_cfg.read_text())
    config["paths"]["output_dir"] = str(out_dir)
    config["paths"]["analysis_dir"] = str(ANA / f"seed{seed}")
    cfg_path = ANA / f"seed{seed}/infer_config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(config, sort_keys=False))

    cmd = ["python3", str(EXP017 / "inference.py"), "--config", str(cfg_path)]
    for ckpt in checkpoints:
        cmd += ["--checkpoint", str(ckpt)]
    print(f"seed{seed}: inference with {len(checkpoints)} checkpoints", flush=True)
    subprocess.run(cmd, check=True)
    return out_dir


def blend(name: str, weights: dict[str, float], sources: dict[str, Path]) -> dict:
    files = {key: {p.name: p for p in (sources[key] / "test_files").glob("*.tif")} for key in weights}
    common = sorted(set.intersection(*(set(v) for v in files.values())))
    if not common:
        return {"name": name, "status": "missing_predictions", "weights": weights}

    dest = SUB / f"exp027/{name}/test_files"
    dest.mkdir(parents=True, exist_ok=True)
    for fn in common:
        ref = files[next(iter(weights))][fn]
        arr = sum(w * read_tiff_array(files[key][fn])[0].astype(np.float32) for key, w in weights.items())
        write_float32_like_template(ref, dest / fn, np.maximum(arr, 0.0))
    shutil.copy2(ROOT / "data/evaluation_dataset/evaluation_target.csv", dest.parent / "evaluation_target.csv")
    return {"name": name, "status": "complete", "weights": weights, "files": len(common)}


def apply_patch(name: str) -> str:
    patch_config = {
        "experiment": {"name": "exp027", "description": f"overlap patch on exp027/{name}", "seed": 42},
        "data": {
            "train_csv": str(ROOT / "data/train_dataset/train_dataset.csv"),
            "evaluation_csv": str(ROOT / "data/evaluation_dataset/evaluation_target.csv"),
            "train_dir": str(ROOT / "data/train_dataset"),
            "evaluation_dir": str(ROOT / "data/evaluation_dataset"),
        },
        "overlap": {"pairs_table": str(EXP014 / "overlap_pairs.csv"), "min_agreement": 0.0},
        "paths": {
            "source_submission_dir": str(SUB / f"exp027/{name}"),
            "output_dir": str(SUB / f"exp027/{name}_patched"),
            "submission_zip": str(SUB / f"exp027_{name}_patched.zip"),
        },
    }
    cfg_path = ANA / f"patch_{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(patch_config, sort_keys=False))
    subprocess.run(
        ["python3", str(EXP014 / "apply_overlap.py"), "--config", str(cfg_path)], check=True
    )
    return str(SUB / f"exp027_{name}_patched.zip")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--force-inference", action="store_true")
    args = parser.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]

    ANA.mkdir(parents=True, exist_ok=True)

    sources: dict[str, Path] = {"exp016": SUB / "exp016", "exp017": SUB / "exp017"}
    for seed in seeds:
        sources[f"exp017_seed{seed}"] = run_seed_inference(seed, args.force_inference)

    seed_keys = [f"exp017_seed{s}" for s in seeds]
    family = ["exp017", *seed_keys]
    schemes = {
        # Keep the LB-winning 50/50 balance between the two model types; spread the
        # exp017 half across its seed replicas.
        "half016_half017family": {"exp016": 0.5, **{k: 0.5 / len(family) for k in family}},
        "equal_all": {k: 1.0 / (1 + len(family)) for k in ["exp016", *family]},
    }

    manifest = []
    for name, weights in schemes.items():
        entry = blend(name, weights, sources)
        if entry["status"] == "complete":
            entry["patched_zip"] = apply_patch(name)
        manifest.append(entry)
        print(json.dumps(entry, indent=2), flush=True)

    (ANA / "analysis_summary.json").write_text(json.dumps({"schemes": manifest}, indent=2))


if __name__ == "__main__":
    main()
