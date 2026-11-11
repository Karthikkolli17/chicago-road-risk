"""Context layers for segment features: schools, hospitals (OSM) and community areas (city portal)."""
from __future__ import annotations

import logging
import sys

import geopandas as gpd
import osmnx as ox
import pandas as pd
from shapely.geometry import shape

from src import config
from src.ingest.osm import OSM_DIR, overpass_retry
from src.ingest.soda import SodaClient

log = logging.getLogger(__name__)

POIS_PATH = OSM_DIR / "pois.parquet"
COMMUNITY_AREAS_PATH = config.DATA / "community_areas.parquet"
COMMUNITY_AREAS_ID = "igwz-8jzy"

# OSM tags each kind of place in more than one way; any match counts.
POI_TAGS = {
    "school": {"amenity": ["school"]},
    "hospital": {"amenity": ["hospital"], "healthcare": ["hospital"]},
}


def classify_poi(row: pd.Series) -> str | None:
    for kind, tags in POI_TAGS.items():
        for key, values in tags.items():
            if row.get(key) in values:
                return kind
    return None


def tidy_pois(raw: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """One row per school/hospital: kind, name, geometry in the project CRS.

    Schools are often mapped both as a campus outline and as a point inside it,
    so points lying inside a same-kind polygon are dropped.
    """
    df = raw.reset_index()
    df["kind"] = df.apply(classify_poi, axis=1)
    df = df[df["kind"].notna() & df.geometry.notna()]
    keep = ["element", "id", "kind", "name", "geometry"]
    df = gpd.GeoDataFrame(df.reindex(columns=keep), geometry="geometry", crs=raw.crs).to_crs(config.CRS)
    df = df[df.geom_type.isin(["Point", "Polygon", "MultiPolygon"])]

    polys = df[df.geom_type != "Point"]
    points = df[df.geom_type == "Point"]
    inside = gpd.sjoin(points, polys[["kind", "geometry"]], predicate="within", how="inner")
    dup = inside.index[inside["kind_left"] == inside["kind_right"]].unique()
    out = df.drop(index=dup).reset_index(drop=True)
    out["osm_id"] = out["element"].astype(str) + "/" + out["id"].astype(str)
    return out[["osm_id", "kind", "name", "geometry"]]


PBF_PATH = OSM_DIR / "illinois-latest.osm.pbf"
PBF_URL = "https://download.geofabrik.de/north-america/us/illinois-latest.osm.pbf"
# Schools just across the city line (Evanston, Oak Park...) are still the nearest
# school for edge-of-city streets, so keep POIs up to this far outside the city.
POI_BUFFER_FT = 1000 * config.FT_PER_M


def read_pois_from_pbf(path) -> gpd.GeoDataFrame:
    """Schools/hospitals from a Geofabrik extract: nodes as points, closed ways and
    multipolygon relations as polygons. Same shape as osmnx's features output."""
    import osmium
    from shapely import wkb

    factory = osmium.geom.WKBFactory()
    rows = []

    def pick(tags):
        if "amenity" not in tags and "healthcare" not in tags:
            return None
        t = {k: tags.get(k) for k in ("amenity", "healthcare", "name")}
        return t if classify_poi(t) else None

    class Handler(osmium.SimpleHandler):
        def node(self, n):
            if (t := pick(n.tags)) is not None:
                rows.append(("node", n.id, t, wkb.loads(factory.create_point(n), hex=True)))

        def area(self, a):
            if (t := pick(a.tags)) is not None:
                try:
                    geom = wkb.loads(factory.create_multipolygon(a), hex=True)
                except RuntimeError:  # broken outline in OSM; skip it
                    return
                rows.append(("way" if a.from_way() else "relation", a.orig_id(), t, geom))

    Handler().apply_file(str(path), locations=True)
    idx = pd.MultiIndex.from_tuples([(e, i) for e, i, _, _ in rows], names=["element", "id"])
    return gpd.GeoDataFrame([t for _, _, t, _ in rows], index=idx,
                            geometry=[g for *_, g in rows], crs="EPSG:4326")


def clip_to_city(pois: gpd.GeoDataFrame, areas: gpd.GeoDataFrame, buffer_ft: float = POI_BUFFER_FT):
    city = areas.union_all().buffer(buffer_ft)
    return pois[pois.intersects(city)].reset_index(drop=True)


def fetch_pois(source: str = "pbf") -> gpd.GeoDataFrame:
    """source='pbf' reads the Geofabrik Illinois extract (download it first, see
    PBF_URL); source='overpass' queries the public Overpass API, which is often full."""
    if source == "pbf":
        if not PBF_PATH.exists():
            raise FileNotFoundError(f"download {PBF_URL} to {PBF_PATH} first")
        raw = read_pois_from_pbf(PBF_PATH)
    else:
        tags: dict[str, list[str]] = {}
        for t in POI_TAGS.values():
            for k, v in t.items():
                tags.setdefault(k, []).extend(v)
        raw = overpass_retry(lambda: ox.features_from_place(config.OSM_PLACE, tags=tags),
                             attempts_per_server=15)
    areas = gpd.read_parquet(COMMUNITY_AREAS_PATH)
    pois = clip_to_city(tidy_pois(raw), areas)
    pois.to_parquet(POIS_PATH, index=False)
    log.info("saved POIs: %s", pois["kind"].value_counts().to_dict())
    return pois


def tidy_community_areas(rows: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf = gpd.GeoDataFrame(
        {
            "area_number": pd.to_numeric(rows["area_numbe"]).astype(int),
            "community": rows["community"].str.title(),
        },
        geometry=[shape(g) for g in rows["the_geom"]],
        crs="EPSG:4326",
    ).to_crs(config.CRS)
    return gdf.sort_values("area_number").reset_index(drop=True)


def fetch_community_areas() -> gpd.GeoDataFrame:
    client = SodaClient(config.SODA_DOMAIN, config.SODA_APP_TOKEN)
    rows = client.fetch(COMMUNITY_AREAS_ID, ["the_geom", "area_numbe", "community"], order="area_numbe")
    gdf = tidy_community_areas(rows)
    if len(gdf) != 77 or gdf["area_number"].tolist() != list(range(1, 78)):
        raise RuntimeError(f"expected community areas 1..77, got {len(gdf)} rows")
    gdf.to_parquet(COMMUNITY_AREAS_PATH, index=False)
    log.info("saved %d community areas", len(gdf))
    return gdf


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fetch_community_areas()
    fetch_pois(sys.argv[1] if len(sys.argv) > 1 else "pbf")
