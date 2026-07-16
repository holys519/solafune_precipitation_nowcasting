#!/usr/bin/env python3
"""H1/H2/E-9 (doc/imerg_physics_notes.md): IMERG time-series structure from train GPM truth.

- H1 two-state hypothesis: innovation (1 - corr with the 30-min-previous frame, and the
  advection-residual version) should be bimodal — "morphed" frames are nearly explained by
  the previous frame, "PMW-fresh" frames jump. Also measures spacing between
  high-innovation frames (PMW revisit signature, expected quasi-periodic 1-3 h).
- H2 advection hypothesis: advected persistence (previous frame shifted by the best
  integer displacement within ±4 px) should beat plain persistence clearly, because IMERG
  itself is built by morphing.
- E-9 diurnal cycle: rain fraction / mean by local solar hour (lon/15 offset, approximate
  coordinates — analysis only, see locations.py header).

Reads every train GPM tile (~40.7k files) with the dependency-free tiff reader.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "g_experiments" / "exp017"))
from tiff_utils import read_tiff_array  # noqa: E402

from locations import APPROX_COORDS  # noqa: E402

TRAIN_CSV = PROJECT_DIR / "data" / "train_dataset" / "train_dataset.csv"
GPM_DIR = PROJECT_DIR / "data" / "train_dataset" / "gpm_imerg"
OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp005"
MAX_SHIFT = 4


def best_shift(prev: np.ndarray, cur: np.ndarray) -> tuple[int, int, float, float]:
    """Best integer (dy, dx) within ±MAX_SHIFT maximizing correlation on the overlap.
    Returns (dy, dx, corr_at_best, corr_at_zero)."""
    best = (-2.0, 0, 0)
    zero_corr = -2.0
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            a = prev[max(0, dy):41 + min(0, dy), max(0, dx):41 + min(0, dx)]
            b = cur[max(0, -dy):41 + min(0, -dy), max(0, -dx):41 + min(0, -dx)]
            sa, sb = a.std(), b.std()
            corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb)) if sa > 0 and sb > 0 else 0.0
            if dy == 0 and dx == 0:
                zero_corr = corr
            if corr > best[0]:
                best = (corr, dy, dx)
    return best[1], best[2], best[0], zero_corr


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    rows = list(csv.DictReader(TRAIN_CSV.open(newline="")))
    by_location: dict[str, list[tuple[datetime, str, str]]] = defaultdict(list)
    for row in rows:
        by_location[row["name_location"]].append(
            (datetime.fromisoformat(row["datetime"]), row["gpm_imerg_filename"],
             row["satellite_target"]))

    pair_rows = []
    lag_corr_sums: dict[tuple[str, int], list[float]] = defaultdict(list)
    diurnal: dict[tuple[str, int], list[float]] = defaultdict(list)
    n_read = 0
    for location, items in sorted(by_location.items()):
        items.sort()
        arrays: dict[datetime, np.ndarray] = {}
        satellite = items[0][2]
        lon = APPROX_COORDS.get(location, (0.0, 0.0))[0]
        for when, filename, _ in items:
            arr, _meta = read_tiff_array(GPM_DIR / filename)
            arrays[when] = arr.astype(np.float32)
            n_read += 1
            hour = int((when.hour + when.minute / 60.0 + lon / 15.0) % 24)
            diurnal[(satellite, hour)].append(float((arr > 0).mean()))
        times = sorted(arrays)
        # lag-k full-field autocorrelation (k in 30-min steps)
        for k in (1, 2, 3, 4):
            for when in times:
                other = when + timedelta(minutes=30 * k)
                if other in arrays:
                    a, b = arrays[when], arrays[other]
                    if a.std() > 0 and b.std() > 0:
                        lag_corr_sums[(satellite, k)].append(
                            float(((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std())))
        # consecutive-pair innovation + advection
        for when in times:
            nxt = when + timedelta(minutes=30)
            if nxt not in arrays:
                continue
            prev, cur = arrays[when], arrays[nxt]
            if prev.std() == 0 or cur.std() == 0:
                continue
            dy, dx, corr_shift, corr_zero = best_shift(prev, cur)
            rmse_persist = float(np.sqrt(np.square(cur - prev).mean()))
            shifted = np.roll(np.roll(prev, dy, axis=0), dx, axis=1)
            rmse_advect = float(np.sqrt(np.square(cur - shifted).mean()))
            pair_rows.append({
                "name_location": location, "satellite": satellite,
                "datetime": nxt.isoformat(),
                "corr_zero": round(corr_zero, 4), "corr_shift": round(corr_shift, 4),
                "shift_dy": dy, "shift_dx": dx,
                "innovation_zero": round(1 - corr_zero, 4),
                "innovation_advect": round(1 - corr_shift, 4),
                "rmse_persist": round(rmse_persist, 4),
                "rmse_advect": round(rmse_advect, 4),
            })
        print(f"{location}: {len(times)} frames done ({time.time()-started:.0f}s)", flush=True)

    with (OUT_DIR / "innovation_pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
        writer.writeheader()
        writer.writerows(pair_rows)

    inn = np.array([r["innovation_advect"] for r in pair_rows])
    report = ["# IMERG innovation / advection / diurnal analysis (g_eda/exp005 H1/H2/E-9)", "",
              f"tiles read: {n_read}, consecutive pairs: {len(pair_rows)}", "",
              "## H2: advection vs persistence (mean over pairs)",
              f"- plain persistence RMSE: {np.mean([r['rmse_persist'] for r in pair_rows]):.4f}",
              f"- advected persistence RMSE: {np.mean([r['rmse_advect'] for r in pair_rows]):.4f}",
              f"- corr gain from shift: {np.mean([r['corr_shift'] - r['corr_zero'] for r in pair_rows]):.4f}",
              f"- pairs with nonzero best shift: "
              f"{np.mean([1.0 if (r['shift_dy'], r['shift_dx']) != (0, 0) else 0.0 for r in pair_rows]):.1%}",
              "", "## H1: innovation distribution (advection-residual, 1-corr)"]
    quantiles = np.quantile(inn, [0.1, 0.25, 0.5, 0.75, 0.9])
    report.append(f"- quantiles 10/25/50/75/90%: {[round(float(q), 3) for q in quantiles]}")
    hist, edges = np.histogram(inn, bins=40, range=(0.0, 2.0))
    with (OUT_DIR / "innovation_hist.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_left", "count"])
        for left, count in zip(edges[:-1], hist):
            writer.writerow([round(float(left), 3), int(count)])
    # spacing between high-innovation frames (top quartile), per location
    threshold = float(np.quantile(inn, 0.75))
    spacings = []
    by_loc_pairs: dict[str, list[dict]] = defaultdict(list)
    for r in pair_rows:
        by_loc_pairs[r["name_location"]].append(r)
    for location, rs in by_loc_pairs.items():
        rs.sort(key=lambda r: r["datetime"])
        last = None
        for r in rs:
            if r["innovation_advect"] >= threshold:
                when = datetime.fromisoformat(r["datetime"])
                if last is not None:
                    gap = (when - last).total_seconds() / 60.0
                    if gap <= 360:
                        spacings.append(gap)
                last = when
    if spacings:
        spacing_hist = {int(gap): 0 for gap in range(30, 361, 30)}
        for gap in spacings:
            key = int(round(gap / 30.0) * 30)
            if key in spacing_hist:
                spacing_hist[key] += 1
        report += ["", "## H1: spacing of high-innovation frames (top-quartile), minutes",
                   f"- {json.dumps(spacing_hist)}",
                   "- PMW revisit signature = mass concentrated at 60-180 min rather than 30 min"]

    report += ["", "## lag-k full-field autocorrelation (30-min steps)"]
    for satellite in ("goes", "himawari", "meteosat"):
        vals = [f"lag{k}={np.mean(lag_corr_sums[(satellite, k)]):.3f}" for k in (1, 2, 3, 4)
                if lag_corr_sums[(satellite, k)]]
        report.append(f"- {satellite}: " + " ".join(vals))

    report += ["", "## E-9: diurnal cycle (rain fraction by local solar hour)"]
    diurnal_rows = []
    for satellite in ("goes", "himawari", "meteosat"):
        series = [(h, float(np.mean(diurnal[(satellite, h)]))) for h in range(24)
                  if diurnal[(satellite, h)]]
        if series:
            values = [v for _, v in series]
            peak_hour = series[int(np.argmax(values))][0]
            report.append(f"- {satellite}: mean {np.mean(values):.3f}, "
                          f"amplitude {max(values) - min(values):.3f}, peak hour {peak_hour}")
            for h, v in series:
                diurnal_rows.append({"satellite": satellite, "local_hour": h,
                                     "rain_fraction": round(v, 4)})
    with (OUT_DIR / "diurnal_cycle.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["satellite", "local_hour", "rain_fraction"])
        writer.writeheader()
        writer.writerows(diurnal_rows)

    (OUT_DIR / "IMERG_INNOVATION.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report), flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
