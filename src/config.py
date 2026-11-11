from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
WAREHOUSE = DATA / "warehouse.duckdb"
REPORTS = ROOT / "reports"

SODA_DOMAIN = "data.cityofchicago.org"
SODA_APP_TOKEN = os.environ.get("SODA_APP_TOKEN")  # optional; raises rate limits

CRASHES_ID = "85ca-t3if"
PEOPLE_ID = "u6pd-qa9d"
START_DATE = "2021-01-01"

# Pinned column lists. The portal adds columns now and then; never use SELECT *.
# Only road-agnostic identifiers + target fields are pulled. Per-crash conditions
# (weather, lighting, crash type) are deliberately excluded to avoid feature leakage.
CRASH_COLUMNS = [
    "crash_record_id",
    "crash_date",
    "posted_speed_limit",
    "most_severe_injury",
    "injuries_fatal",
    "injuries_incapacitating",
    "latitude",
    "longitude",
    # location only: used to pick the right road at intersections, never as a model feature
    "street_direction",
    "street_name",
]
PEOPLE_COLUMNS = [
    "person_id",
    "crash_record_id",
    "crash_date",
    "person_type",
    "injury_classification",
]
VRU_PERSON_TYPES = ["PEDESTRIAN", "BICYCLE"]  # vulnerable road users we flag
KSI_INJURIES = ["FATAL", "INCAPACITATING INJURY"]

# Generous bounding box around the City of Chicago (incl. O'Hare to the west).
CITY_BBOX = {"lat_min": 41.64, "lat_max": 42.03, "lon_min": -87.95, "lon_max": -87.52}

OSM_PLACE = "Chicago, Illinois, USA"
EXCLUDED_HIGHWAYS = {"motorway", "motorway_link"}  # CPD does not cover these
KEEP_MOTORWAY_NAMES = ("Lake Shore",)  # tagged motorway in OSM, but a city road CPD patrols
NETWORK_BUFFER_M = 100  # grow the city boundary so border streets are in the network
CRS = "EPSG:3435"  # NAD83 / Illinois East — NOTE: units are US survey FEET
FT_PER_M = 1 / 0.3048006096
SNAP_MAX_FT = 30 * FT_PER_M  # the 30 m snapping cap, in CRS units
