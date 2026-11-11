from __future__ import annotations

import logging
import time

import geopandas as gpd
import osmnx as ox
import pandas as pd
import requests

from src import config

log = logging.getLogger(__name__)

OSM_DIR = config.DATA / "osm"
SEGMENTS_PATH = OSM_DIR / "segments.parquet"
NODES_PATH = OSM_DIR / "nodes.parquet"


def _as_list(value) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [value]


def is_excluded(highway) -> bool:
    """True if every highway tag on the edge is a motorway type. Edges OSM merged
    from e.g. ['motorway_link', 'primary'] are kept, since CPD covers part of them."""
    tags = _as_list(highway)
    return bool(tags) and all(t in config.EXCLUDED_HIGHWAYS for t in tags)


def _flatten(value):
    """Parquet needs one type per column, but OSM tags mix scalars and lists
    (e.g. osmid 123 vs [123, 456]), so every value becomes text and lists become 'a;b'."""
    if isinstance(value, list):
        return ";".join(str(v) for v in value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return str(value)


OVERPASS_URLS = ["https://overpass-api.de/api", "https://overpass.kumi.systems/api"]


def configure_osmnx() -> None:
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(config.ROOT / "cache")
    ox.settings.requests_timeout = 300


def overpass_retry(download, attempts_per_server: int = 4, max_wait: int = 600, sleep=time.sleep):
    """Run `download()` against the public Overpass servers, which turn requests away
    when busy: back off and retry, then try a mirror. Finished sub-queries are
    cached, so retries resume."""
    configure_osmnx()
    last_exc = None
    for url in OVERPASS_URLS:
        ox.settings.overpass_url = url
        for attempt in range(attempts_per_server):
            try:
                log.info("querying Overpass at %s", url)
                return download()
            except (ox._errors.ResponseStatusCodeError, requests.exceptions.RequestException) as exc:
                last_exc = exc
                wait = min(30 * 2**attempt, max_wait)
                log.warning("Overpass busy (%s); retrying in %ds", str(exc).splitlines()[0], wait)
                sleep(wait)
    raise last_exc


def build_network():
    G = overpass_retry(lambda: ox.graph_from_place(config.OSM_PLACE, network_type="drive"))
    n_before = G.number_of_edges()

    drop = [(u, v, k) for u, v, k, hw in G.edges(keys=True, data="highway") if is_excluded(hw)]
    G.remove_edges_from(drop)
    G.remove_nodes_from([n for n, d in G.degree() if d == 0])
    log.info("removed %d motorway edges of %d", len(drop), n_before)

    # A two-way street is two directed edges; one physical segment should be one row.
    G = ox.convert.to_undirected(G)
    G = ox.project_graph(G, to_crs=config.CRS)
    nodes, edges = ox.graph_to_gdfs(G)

    edges = edges.reset_index()
    edges.insert(0, "segment_id", edges["u"].astype(str) + "-" + edges["v"].astype(str) + "-" + edges["key"].astype(str))
    for col in edges.columns:
        if edges[col].dtype == object and col != "geometry":
            edges[col] = edges[col].map(_flatten)
    nodes = nodes.reset_index()

    OSM_DIR.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(SEGMENTS_PATH, index=False)
    nodes.to_parquet(NODES_PATH, index=False)
    log.info("saved %d segments, %d nodes in %s", len(edges), len(nodes), config.CRS)
    return nodes, edges


def load_segments() -> gpd.GeoDataFrame:
    return gpd.read_parquet(SEGMENTS_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_network()
