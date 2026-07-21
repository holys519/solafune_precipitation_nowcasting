"""Closed-form position/time features from a frozen (lat, lon) table + each row's own UTC
datetime. Officially permitted per discussion/geocoding_coordinates_ja.md and
discussion/approved_geocoding_sources_ja.md: "latitude/longitude, hemisphere, sin/cos
position encodings, solar geometry, local solar time" are allowed; only DEM/coastline/
climate-map joins (external datasets) are not. No network calls happen here -- coordinates
are loaded from g_eda/exp005/geocoded_locations.csv, a table frozen once via Nominatim
(g_eda/exp005/run_geocode_locations.py) with the source query, URL, and retrieval time
recorded for reproducibility.

Feature set is intentionally small (5 scalars) and tied directly to the one validated
finding in doc/imerg_physics_notes.md (E-9: diurnal cycle amplitude confirmed above noise,
GOES-footprint 16:00 local peak, Himawari-footprint 04:00 peak) rather than every possible
position feature -- raw latitude/longitude fed to a high-capacity model risks memorizing
train-region climate since train/eval locations never overlap (see the same doc's warning).
"""

from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path


def load_geocoded_locations(path: Path) -> dict[str, tuple[float, float]]:
    table: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            table[row["name_location"]] = (float(row["latitude"]), float(row["longitude"]))
    return table


def equation_of_time_minutes(day_of_year: int) -> float:
    """Standard closed-form approximation (Spencer-type Fourier fit), no external data."""
    b = 2 * math.pi * (day_of_year - 81) / 364.0
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)


def local_solar_time_hours(dt_utc: datetime, longitude: float) -> float:
    day_of_year = dt_utc.timetuple().tm_yday
    eot = equation_of_time_minutes(day_of_year)
    utc_hours = dt_utc.hour + dt_utc.minute / 60.0 + dt_utc.second / 3600.0
    lst = utc_hours + longitude / 15.0 + eot / 60.0
    return lst % 24.0


def solar_position_channels(lat: float, lon: float, dt_utc: datetime) -> tuple[float, float, float, float, float]:
    """Returns (lst_sin, lst_cos, hemisphere, doy_sin, doy_cos)."""
    day_of_year = dt_utc.timetuple().tm_yday
    doy_frac = 2 * math.pi * (day_of_year - 1) / 365.0
    lst = local_solar_time_hours(dt_utc, lon)
    lst_frac = 2 * math.pi * lst / 24.0
    return (
        math.sin(lst_frac),
        math.cos(lst_frac),
        1.0 if lat >= 0 else -1.0,
        math.sin(doy_frac),
        math.cos(doy_frac),
    )


SOLAR_FEATURE_NAMES = ("lst_sin", "lst_cos", "hemisphere", "doy_sin", "doy_cos")
N_SOLAR_CHANNELS = len(SOLAR_FEATURE_NAMES)
