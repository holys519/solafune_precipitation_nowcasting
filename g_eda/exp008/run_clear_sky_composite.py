#!/usr/bin/env python3
"""g_eda/exp008: clear-sky composites via temporal percentile stacking (community EDA idea
from the Solafune discussion, 2026-07-20). Removes cloud cover per-pixel using percentile
compositing over many frames: a low percentile of reflectance (clouds are bright, ground is
dark) gives a clear-sky "true color" surface view; a high percentile of IR-window brightness
temperature (cloud tops are cold, ground is warm) gives a clear-sky surface temperature map.

Used here purely as EDA, to sanity-check whether locations we'd flagged as anomalous have an
obvious terrain explanation:
- bihar / dhaka (shafimiakhil's reported dryness, doc/discussion/round6_findings) -- water
  body, floodplain, or urban-heat-island signature that could explain suppressed rainfall?
- upper_midwest / rio_grande_do_sul (GOES 4-band corruption concentration) -- any terrain
  correlation with why these two eval locations are affected?

Only uses this competition's own provided imagery (dataset-wide percentile statistics over
many past frames of the SAME location) -- no external data. This is NOT wired into any model
input. If it were turned into a model feature later, it would need a strictly causal,
per-row expanding-window rebuild (only frames with timestamp <= T-10 for that row) to stay
compliant with the 2026-07-20 causality ruling -- an unfiltered whole-dataset composite like
this script builds would repeat the exact successor-row mistake for eval locations.
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "g_experiments" / "exp038"))
from tiff_utils import read_tiff_array  # noqa: E402

# 0-indexed band positions (community post's CFG table). Cross-checked against our own
# KEY_BANDS table (g_experiments/exp045/dataset.py) -- vis/red and win/ir indices match
# exactly for all three satellites, a good independent confirmation of that mapping.
CFG = {
    "himawari": dict(red=2, grn=1, blu=0, nir=3, swir=4, ir=12, veggie=None),
    "goes":     dict(red=1, grn=None, blu=0, nir=2, swir=4, ir=12, veggie=2),
    "meteosat": dict(red=2, grn=1, blu=0, nir=3, swir=6, ir=13, veggie=None),
}
SATELLITE_LABEL = {"himawari": "Himawari", "goes": "GOES", "meteosat": "Meteosat"}

DAY_RED_MEAN_THRESHOLD = 25.0


def stretch(x: np.ndarray, lo: float = 2.0, hi: float = 98.0) -> np.ndarray:
    a, b = np.percentile(x, [lo, hi])
    return np.clip((x - a) / (b - a + 1e-6), 0.0, 1.0)


def to_uint8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0.0, 1.0) * 255).astype(np.uint8)


def build_composite(files: list[Path], cfg: dict) -> tuple[np.ndarray, np.ndarray, int, int]:
    vis_stack: list[np.ndarray] = []
    ir_stack: list[np.ndarray] = []
    skipped = 0
    for f in files:
        arr, _meta = read_tiff_array(f)
        if arr.ndim != 3 or arr.shape[-1] < 16:
            skipped += 1
            continue
        ir_stack.append(arr[:, :, cfg["ir"]].astype(np.float32))
        if arr[:, :, cfg["red"]].astype(np.float32).mean() > DAY_RED_MEAN_THRESHOLD:
            vis_stack.append(arr[:, :, :8].astype(np.float32))
    if not vis_stack or not ir_stack:
        raise ValueError("no usable frames (all defective, or no daytime frames found)")
    bt_clear = np.percentile(np.stack(ir_stack), 98, axis=0)
    vis_clear = np.percentile(np.stack(vis_stack), 10, axis=0)
    return vis_clear, bt_clear, len(vis_stack), skipped


def save_panels(site: str, sat: str, vis_clear: np.ndarray, bt_clear: np.ndarray, cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    R, B = vis_clear[:, :, cfg["red"]], vis_clear[:, :, cfg["blu"]]
    if cfg["grn"] is not None:
        G = vis_clear[:, :, cfg["grn"]]
    else:
        G = 0.45 * R + 0.45 * B + 0.1 * vis_clear[:, :, cfg["veggie"]]
    true_color = np.dstack([to_uint8(stretch(x)) for x in (R, G, B)])
    natural = np.dstack([to_uint8(stretch(vis_clear[:, :, i])) for i in (cfg["swir"], cfg["nir"], cfg["red"])])
    ndvi = (vis_clear[:, :, cfg["nir"]] - R) / (vis_clear[:, :, cfg["nir"]] + R + 1e-6)
    ndvi_img = to_uint8(np.clip((ndvi + 0.3) / 0.8, 0.0, 1.0))
    bt_img = to_uint8(stretch(bt_clear))

    Image.fromarray(true_color).save(out_dir / f"{site}_{sat}_true_color.png")
    Image.fromarray(natural).save(out_dir / f"{site}_{sat}_natural_color.png")
    Image.fromarray(ndvi_img).save(out_dir / f"{site}_{sat}_ndvi.png")
    Image.fromarray(bt_img).save(out_dir / f"{site}_{sat}_bt.png")

    water_frac = float((ndvi < 0).mean())
    veg_frac = float((ndvi > 0.2).mean())
    print(f"{site}/{sat}: water(NDVI<0)={water_frac:.3f} veg(NDVI>0.2)={veg_frac:.3f} "
          f"BT_raw=[{bt_clear.min():.1f},{bt_clear.max():.1f}] mean={bt_clear.mean():.1f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True)
    parser.add_argument("--satellite", required=True, choices=list(CFG))
    parser.add_argument("--data-dir", required=True,
                         help="e.g. ../../data/train_dataset or ../../data/evaluation_dataset")
    parser.add_argument("--out-dir", default=str(ROOT / "outputs" / "g_eda" / "exp008"))
    args = parser.parse_args()

    cfg = CFG[args.satellite]
    data_dir = Path(args.data_dir)
    sat_dir = data_dir / args.satellite
    prefix = "train" if data_dir.name == "train_dataset" else "test"
    pattern = str(sat_dir / f"{prefix}_{args.site}_{SATELLITE_LABEL[args.satellite]}_*.tif")
    files = sorted(glob.glob(pattern))[::3]  # thin 10-min cadence to 30 min, as in the original post
    if not files:
        raise FileNotFoundError(f"no files matched {pattern}")

    vis_clear, bt_clear, n_vis, n_skipped = build_composite([Path(f) for f in files], cfg)
    print(f"{args.site}/{args.satellite}: {len(files)} files thinned, {n_vis} daytime frames used, "
          f"{n_skipped} defective frames skipped")
    save_panels(args.site, args.satellite, vis_clear, bt_clear, cfg, Path(args.out_dir))


if __name__ == "__main__":
    main()
