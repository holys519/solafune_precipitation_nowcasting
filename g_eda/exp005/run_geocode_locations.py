#!/usr/bin/env python3
"""Geocode every name_location in train_dataset.csv + evaluation_target.csv via Nominatim
(OpenStreetMap), the organizers' approved source (discussion/approved_geocoding_sources_ja.md).

This produces the FIXED, documented table required before any coordinate-derived feature can
enter a submission pipeline (per discussion/geocoding_coordinates_ja.md's reproducibility
requirement): name_location, source, search query, source record id/url, canonical display
name, latitude, longitude (EPSG:4326), retrieval timestamp. The API is called once here and
the result is frozen to CSV/JSON -- training/inference code must load the frozen table, never
call Nominatim live.

g_eda/exp005/locations.py's APPROX_COORDS predates this and is explicitly documented there as
"OFFLINE ANALYSIS ONLY, re-derive from GeoNames/Nominatim before any submission feature" --
this script is that re-derivation. Query strings add country/region context only where it is
objectively derivable from data already in our own CSVs (the satellite_target column fixes
which continent a GOES/Himawari/Meteosat location must be on); they never guess a specific
city for a vague regional name.

Nominatim usage policy: max 1 request/second, identify with a real User-Agent. 37 locations,
one-time lookup -- well within the policy for ad hoc use (not a live service).
"""

from __future__ import annotations

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_CSV = ROOT / "data" / "train_dataset" / "train_dataset.csv"
EVAL_CSV = ROOT / "data" / "evaluation_dataset" / "evaluation_target.csv"
OUT_CSV = Path(__file__).resolve().parent / "geocoded_locations.csv"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "solafune-precip-nowcasting-research/1.0 (offline one-time geocoding lookup)"

# Query string per location. Country/region qualifiers are added only where derivable from our
# own data (satellite_target's continent) or where the bare name is genuinely ambiguous on its
# own (e.g. "valencia" without a country matches multiple cities worldwide).
QUERIES: dict[str, str] = {
    "aceh": "Aceh, Indonesia",
    "andalusia": "Andalusia, Spain",
    "atlantic_coast": "South Carolina, United States",
    "bahia_blanca": "Bahia Blanca, Argentina",
    "bihar": "Bihar, India",
    "borno_state": "Borno State, Nigeria",
    "cape_town": "Cape Town, South Africa",
    "central_philippines": "Visayas, Philippines",
    "central_vietnam": "Da Nang, Vietnam",
    "dhaka": "Dhaka, Bangladesh",
    "ecuador": "Ecuador",
    "florida": "Florida, United States",
    "france": "France",
    "friuli_venezia_giulia": "Friuli-Venezia Giulia, Italy",
    "gaza_province": "Gaza Province, Mozambique",
    "guangdong": "Guangdong, China",
    "hat_yai": "Hat Yai, Thailand",
    "jakarta": "Jakarta, Indonesia",
    "jamaica": "Jamaica",
    "kanto_region": "Kanto region, Japan",
    "kinshasa": "Kinshasa, Democratic Republic of the Congo",
    "limpopo_province": "Limpopo Province, South Africa",
    "lombardia": "Lombardia, Italy",
    "maputo_province": "Maputo Province, Mozambique",
    "mekong_delta": "Mekong Delta, Vietnam",
    "mexico": "Mexico",
    "niger_state": "Niger State, Nigeria",
    "north_sumatra": "North Sumatra, Indonesia",
    "northeast_malaysia": "Kelantan, Malaysia",
    "peru": "Peru",
    "quang_nam": "Quang Nam Province, Vietnam",
    "rio_grande_do_sul": "Rio Grande do Sul, Brazil",
    "sofala_province": "Sofala Province, Mozambique",
    "sri_lanka": "Sri Lanka",
    "sylhet": "Sylhet, Bangladesh",
    "tanganyika": "Tanganyika",
    "upper_midwest": "Minnesota, United States",
    "valencia": "Valencia, Spain",
}

# Names that are not official gazetteer entries (descriptive/regional, not a place with a
# single unambiguous record) -- flagged so downstream users know these hits are approximate.
FUZZY_NAMES = {
    "atlantic_coast", "central_philippines", "central_vietnam", "northeast_malaysia",
    "upper_midwest", "tanganyika",
}


def geocode(query: str) -> dict:
    params = urllib.parse.urlencode({"q": query, "format": "json", "limit": 1})
    url = f"{NOMINATIM_URL}?{params}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=15) as response:
        results = json.loads(response.read().decode("utf-8"))
    if not results:
        raise ValueError(f"no Nominatim result for query={query!r} (url={url})")
    return {"url": url, **results[0]}


def main() -> None:
    with TRAIN_CSV.open(newline="") as f:
        train_locs = {r["name_location"] for r in csv.DictReader(f)}
    with EVAL_CSV.open(newline="") as f:
        eval_locs = {r["name_location"] for r in csv.DictReader(f)}
    all_locs = sorted(train_locs | eval_locs)

    missing_queries = [loc for loc in all_locs if loc not in QUERIES]
    if missing_queries:
        raise ValueError(f"no query string defined for: {missing_queries}")

    rows = []
    for i, loc in enumerate(all_locs):
        query = QUERIES[loc]
        result = geocode(query)
        rows.append({
            "name_location": loc,
            "split": "train" if loc in train_locs else "",
            "split_eval": "eval" if loc in eval_locs else "",
            "source": "Nominatim",
            "query": query,
            "source_url": result["url"],
            "osm_id": result.get("osm_id", ""),
            "canonical_name": result.get("display_name", ""),
            "latitude": result["lat"],
            "longitude": result["lon"],
            "epsg": "4326",
            "fuzzy_query": loc in FUZZY_NAMES,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"[{i + 1}/{len(all_locs)}] {loc}: ({result['lat']}, {result['lon']}) "
              f"-- {result.get('display_name', '')[:70]}", flush=True)
        time.sleep(1.1)  # Nominatim usage policy: max 1 req/sec

    fieldnames = list(rows[0].keys())
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {OUT_CSV} ({len(rows)} locations, {sum(r['fuzzy_query'] for r in rows)} fuzzy)")


if __name__ == "__main__":
    main()
