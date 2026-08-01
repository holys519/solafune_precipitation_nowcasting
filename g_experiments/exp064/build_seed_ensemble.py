#!/usr/bin/env python3
"""Build a seed-ensemble submission by averaging the per-tile eval predictions of two or more
members of the same architecture, differing only in training seed. Copied from
g_experiments/exp056/build_seed_ensemble.py (identical mechanics), used here for exp064_effb3's
seed42/seed456 pair -- variance-reduction ensemble, same low-inversion-risk category as exp056's
seed ensembles (see doc/oof_lb_transfer_by_category...).

Averages the 29090 float32 GeoTIFFs tile-by-tile, clips at 0, and writes each with the first
member's tile as the metadata template (write_float32_like_template). Then zips
evaluation_target.csv + test_files/ in the standard submission layout.

Usage:
    python3 build_seed_ensemble.py --members exp064_effb3 exp064_effb3_seed456 --name exp064_effb3_seed_ens_42_456
"""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import numpy as np

from tiff_utils import read_tiff_array, write_float32_like_template

SCRIPT_DIR = Path(__file__).resolve().parent
SUBM = (SCRIPT_DIR / "../../outputs/submissions").resolve()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", default=["exp056", "exp056_seed123", "exp056_seed456"])
    ap.add_argument("--name", default=None, help="output basename (default derived from members)")
    args = ap.parse_args()

    members = args.members
    ref = SUBM / members[0]  # exp056 = template + canonical file list + evaluation_target.csv
    ref_files = ref / "test_files"
    filenames = sorted(p.name for p in ref_files.glob("*.tif"))
    if not filenames:
        raise SystemExit(f"no tif under {ref_files}")
    for m in members:
        d = SUBM / m / "test_files"
        n = len(list(d.glob("*.tif")))
        if n != len(filenames):
            raise SystemExit(f"member {m} has {n} tif, expected {len(filenames)}")
    print(f"averaging {len(members)} members over {len(filenames)} tiles: {members}", flush=True)

    name = args.name or ("exp056_seed_ensemble" if len(members) == 3 else "exp056_seed_ens_" + "_".join(
        m.replace("exp056", "s42").replace("_seed", "s") for m in members))
    out_dir = SUBM / name
    out_files = out_dir / "test_files"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_files.mkdir(parents=True)
    shutil.copy(ref / "evaluation_target.csv", out_dir / "evaluation_target.csv")

    member_dirs = [SUBM / m / "test_files" for m in members]
    max_abs_delta_vs_seed42 = 0.0
    for i, fn in enumerate(filenames):
        acc = None
        for d in member_dirs:
            arr, _ = read_tiff_array(d / fn)
            arr = np.asarray(arr, dtype=np.float64)
            acc = arr if acc is None else acc + arr
        mean = np.clip(acc / len(member_dirs), 0.0, None).astype(np.float32)
        # track how far the ensemble moves from seed42 (sanity: should be modest)
        s42, _ = read_tiff_array(member_dirs[0] / fn)
        max_abs_delta_vs_seed42 = max(max_abs_delta_vs_seed42, float(np.abs(mean - np.asarray(s42, np.float32)).max()))
        write_float32_like_template(ref_files / fn, out_files / fn, mean)
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(filenames)}", flush=True)

    zip_path = SUBM / f"{name}_submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(out_dir / "evaluation_target.csv", "evaluation_target.csv")
        for fn in filenames:
            z.write(out_files / fn, f"test_files/{fn}")
    print(f"wrote {zip_path}  (max |ensemble - seed42| = {max_abs_delta_vs_seed42:.4f})", flush=True)


if __name__ == "__main__":
    main()
