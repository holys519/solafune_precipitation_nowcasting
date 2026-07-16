"""Approximate location coordinates for physics EDA (g_eda/exp005).

NOTE ON RULES: the GeoTIFFs' geo-transform was deliberately stripped by the organizers
(identity matrix — verified 2026-07-16), and the official ruling allows deriving
coordinates from location names ONLY via GeoNames or Nominatim, documented per
submission (see discussion/geocoding_coordinates_ja.md and
discussion/approved_geocoding_sources_ja.md).

The values below are approximate region centers used for OFFLINE ANALYSIS ONLY
(hypothesis testing needs ~1-degree accuracy; parallax direction changes slowly with
position). Before any coordinate-derived feature enters a SUBMISSION pipeline, re-derive
these from GeoNames/Nominatim and record the source in the submission description.
"""

# location -> (lon_deg_east, lat_deg_north)
APPROX_COORDS = {
    # train
    "aceh": (95.5, 4.5),
    "andalusia": (-4.5, 37.5),
    "atlantic_coast": (-81.0, 30.5),
    "bahia_blanca": (-62.3, -38.7),
    "bihar": (85.5, 25.5),
    "borno_state": (13.0, 11.5),
    "cape_town": (18.5, -33.9),
    "central_philippines": (123.5, 11.0),
    "central_vietnam": (108.0, 16.0),
    "dhaka": (90.4, 23.8),
    "ecuador": (-78.5, -1.5),
    "florida": (-81.5, 28.0),
    "france": (2.5, 46.5),
    "friuli_venezia_giulia": (13.0, 46.0),
    "gaza_province": (33.5, -23.5),      # Mozambique
    "guangdong": (113.5, 23.0),
    "hat_yai": (100.5, 7.0),
    "jakarta": (106.8, -6.2),
    "jamaica": (-77.3, 18.1),
    "kinshasa": (15.3, -4.3),
    # evaluation
    "kanto_region": (139.7, 35.7),
    "limpopo_province": (29.5, -23.5),
    "lombardia": (9.9, 45.6),
    "maputo_province": (32.5, -25.5),
    "mekong_delta": (105.8, 10.0),
    "mexico": (-99.1, 19.4),
    "niger_state": (6.0, 9.5),
    "north_sumatra": (98.7, 2.5),
    "northeast_malaysia": (102.5, 5.5),
    "peru": (-76.0, -10.0),
    "quang_nam": (108.0, 15.6),
    "rio_grande_do_sul": (-53.0, -30.0),
    "sofala_province": (34.8, -19.5),
    "sri_lanka": (80.7, 7.9),
    "sylhet": (91.9, 24.9),
    "tanganyika": (27.9, -6.3),
    "upper_midwest": (-93.0, 44.0),
    "valencia": (-0.4, 39.5),
}

# Geostationary sub-satellite longitudes. Meteosat could be 0.0 (prime) or 45.5 (IODC);
# the geometry-vs-empirical fit in run_parallax_geometry.py discriminates.
SUBPOINT_CANDIDATES = {
    "goes": (-75.2,),
    "himawari": (140.7,),
    "meteosat": (0.0, 45.5),
}
