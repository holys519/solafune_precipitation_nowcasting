#!/usr/bin/env python3
"""H3 (doc/imerg_physics_notes.md): validate the geostationary parallax geometry against the
empirical per-location shifts, then extrapolate registration shifts to EVAL locations.

Geometry: a cloud at height h viewed from a geostationary satellite appears displaced away
from the sub-satellite point by ~ h * tan(local viewing zenith angle), along the azimuth
from the sub-satellite point to the location. Direction and tan(zenith) are closed-form
functions of (location lat/lon, sub-satellite lon) — an allowed feature transformation.

Fit: empirical (median_dy, median_dx) from outputs/g_eda/exp001/parallax_shift_by_location.csv
regressed on the predicted displacement direction with one scale factor per satellite (the
scale absorbs mean cloud height, grid spacing, and the shift-sign convention of the
empirical estimator). Output: fit quality per satellite + predicted shifts for the 18 eval
locations (usable as an input-registration prior where truth cannot be measured).

Pure CPU/stdlib+numpy.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

from locations import APPROX_COORDS, SUBPOINT_CANDIDATES

PROJECT_DIR = Path(__file__).resolve().parents[2]
EMPIRICAL = PROJECT_DIR / "outputs" / "g_eda" / "exp001" / "parallax_shift_by_location.csv"
OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp005"
TRAIN_CSV = PROJECT_DIR / "data" / "train_dataset" / "train_dataset.csv"
EVAL_CSV = PROJECT_DIR / "data" / "evaluation_dataset" / "evaluation_target.csv"

GEO_RADIUS = 42164.0  # km, geostationary orbit radius
EARTH_RADIUS = 6371.0


def view_geometry(lat: float, lon: float, subpoint_lon: float) -> tuple[float, float, float]:
    """Returns (tan_zenith, u_dy, u_dx): viewing-zenith tangent and the unit displacement
    direction in pixel coordinates (dy = +south/row, dx = +east/col) pointing AWAY from the
    sub-satellite point along the great-circle azimuth."""
    lat_r, dlon_r = math.radians(lat), math.radians(lon - subpoint_lon)
    cos_c = math.cos(lat_r) * math.cos(dlon_r)  # angular distance from sub-point
    c = math.acos(max(-1.0, min(1.0, cos_c)))
    # zenith angle at the ground site for a geostationary satellite
    sin_zen = GEO_RADIUS * math.sin(c) / math.sqrt(
        GEO_RADIUS**2 + EARTH_RADIUS**2 - 2 * GEO_RADIUS * EARTH_RADIUS * cos_c
    )
    zen = math.asin(max(-1.0, min(1.0, sin_zen)))
    # azimuth of the displacement = direction from sub-point toward the location,
    # evaluated at the location: east component ~ sin(dlon)*... use the bearing FROM the
    # location TOWARD the sub-point, then flip (displacement is away from the sub-point).
    y = math.sin(-dlon_r) * math.cos(0.0)  # sub-point latitude is 0
    x = math.cos(lat_r) * math.sin(0.0) - math.sin(lat_r) * math.cos(0.0) * math.cos(-dlon_r)
    bearing_to_subpoint = math.atan2(y, x)  # from north, clockwise, at the location
    away = bearing_to_subpoint + math.pi
    u_north, u_east = math.cos(away), math.sin(away)
    return math.tan(zen), -u_north, u_east  # dy is +south


def satellite_of() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for path in (TRAIN_CSV, EVAL_CSV):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                mapping[row["name_location"]] = row["satellite_target"]
    return mapping


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sat_of = satellite_of()
    empirical = {}
    with EMPIRICAL.open(newline="") as f:
        for row in csv.DictReader(f):
            empirical[row["name_location"]] = (float(row["median_dy"]), float(row["median_dx"]),
                                               int(row["samples"]))

    report = ["# Parallax geometry vs empirical shifts (g_eda/exp005 H3)", ""]
    fits: dict[str, dict] = {}
    for satellite in ("goes", "himawari", "meteosat"):
        best = None
        for subpoint in SUBPOINT_CANDIDATES[satellite]:
            xs, ys = [], []  # predicted direction*tan_zen, empirical vector
            rows = []
            for loc, (dy, dx, n) in empirical.items():
                if sat_of.get(loc) != satellite or loc not in APPROX_COORDS:
                    continue
                lon, lat = APPROX_COORDS[loc]
                tan_zen, u_dy, u_dx = view_geometry(lat, lon, subpoint)
                xs.append([tan_zen * u_dy, tan_zen * u_dx])
                ys.append([dy, dx])
                rows.append((loc, tan_zen, u_dy, u_dx, dy, dx, n))
            x = np.asarray(xs).ravel()
            yv = np.asarray(ys).ravel()
            scale = float(np.dot(x, yv) / np.dot(x, x)) if np.dot(x, x) > 0 else 0.0
            resid = yv - scale * x
            ss_tot = float(np.square(yv - yv.mean()).sum())
            r2 = 1.0 - float(np.square(resid).sum()) / ss_tot if ss_tot else float("nan")
            cosine = float(np.dot(x, yv) / (np.linalg.norm(x) * np.linalg.norm(yv) + 1e-12))
            cand = {"subpoint": subpoint, "scale": scale, "r2": r2, "cosine": cosine,
                    "locations": len(rows), "rows": rows}
            if best is None or abs(cand["cosine"]) > abs(best["cosine"]):
                best = cand
        fits[satellite] = best
        report += [f"## {satellite} (sub-point {best['subpoint']}°E, n={best['locations']})",
                   f"- fitted scale (px per tan(zenith)): {best['scale']:.3f}",
                   f"- direction cosine (pred vs empirical): {best['cosine']:.3f}, R²={best['r2']:.3f}",
                   ""]
        for loc, tan_zen, u_dy, u_dx, dy, dx, n in best["rows"]:
            pred_dy, pred_dx = best["scale"] * tan_zen * u_dy, best["scale"] * tan_zen * u_dx
            report.append(f"  - {loc}: empirical ({dy:+.1f},{dx:+.1f}) vs geometry "
                          f"({pred_dy:+.1f},{pred_dx:+.1f}) [tan_zen={tan_zen:.2f}, n={n}]")
        report.append("")

    # extrapolate to eval locations with the fitted per-satellite scale
    eval_rows = []
    for loc, satellite in sorted(sat_of.items()):
        if loc in empirical or loc not in APPROX_COORDS:
            continue
        fit = fits[satellite]
        lon, lat = APPROX_COORDS[loc]
        tan_zen, u_dy, u_dx = view_geometry(lat, lon, fit["subpoint"])
        eval_rows.append({"name_location": loc, "satellite_target": satellite,
                          "tan_zenith": round(tan_zen, 4),
                          "pred_dy": round(fit["scale"] * tan_zen * u_dy, 2),
                          "pred_dx": round(fit["scale"] * tan_zen * u_dx, 2),
                          "fit_cosine": round(fit["cosine"], 3)})
    with (OUT_DIR / "eval_parallax_prior.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(eval_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_rows)
    report += ["## Eval-location registration priors (eval_parallax_prior.csv)", ""]
    for row in eval_rows:
        report.append(f"  - {row['name_location']} ({row['satellite_target']}): "
                      f"pred (dy,dx)=({row['pred_dy']:+.1f},{row['pred_dx']:+.1f})")
    report += ["", "Verdict rule: only trust the extrapolation for satellites with "
               "|direction cosine| > 0.6; otherwise the empirical shifts are not "
               "parallax-dominated and registration should stay off for that satellite."]
    (OUT_DIR / "PARALLAX_GEOMETRY.md").write_text("\n".join(report), encoding="utf-8")
    (OUT_DIR / "parallax_fits.json").write_text(
        json.dumps({sat: {k: v for k, v in fit.items() if k != "rows"} for sat, fit in fits.items()},
                   indent=2), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
