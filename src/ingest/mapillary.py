from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import mapbox_vector_tile as mvt
import pandas as pd
import requests
import shapely
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from src import config
from src.ingest.context import COMMUNITY_AREAS_PATH
from src.ingest.soda import _is_retryable

log = logging.getLogger(__name__)

MLY_DIR = config.DATA / "mapillary"
TILE_DIR = MLY_DIR / "tiles"
DETECTIONS_PATH = MLY_DIR / "detections.parquet"
IMAGES_PATH = MLY_DIR / "images.parquet"

ZOOM = 14  # the only zoom where the detection layers have every point
TILE_URL = "https://tiles.mapillary.com/maps/vtp/{layer}/2/{z}/{x}/{y}"
LAYERS = {"points": "mly_map_feature_point", "signs": "mly_map_feature_traffic_sign", "coverage": "mly1_public"}
WORKERS = 8


def token() -> str:
    tok = os.environ.get("MAPILLARY_TOKEN")
    env = config.ROOT / ".env"
    if not tok and env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("MAPILLARY_TOKEN="):
                tok = line.split("=", 1)[1].strip().strip("'\"")
    if not tok or not tok.startswith("MLY|"):
        raise RuntimeError("MAPILLARY_TOKEN missing: put MAPILLARY_TOKEN=MLY|... in .env")
    return tok


# tiles
def lonlat_to_tile(lon: float, lat: float, z: int = ZOOM) -> tuple[int, int]:
    n = 2 ** z
    x = int((lon + 180) / 360 * n)
    y = int((1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)
    return x, y


def tile_bounds(x: int, y: int, z: int = ZOOM) -> tuple[float, float, float, float]:

    n = 2 ** z
    lat = lambda t: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * t / n))))  # noqa: E731
    return x / n * 360 - 180, lat(y + 1), (x + 1) / n * 360 - 180, lat(y)


def city_tiles(z: int = ZOOM) -> list[tuple[int, int]]:

    city = gpd.read_parquet(COMMUNITY_AREAS_PATH).to_crs("EPSG:4326").union_all()
    lon0, lat0, lon1, lat1 = city.bounds
    x0, y0 = lonlat_to_tile(lon0, lat1, z)
    x1, y1 = lonlat_to_tile(lon1, lat0, z)
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)
            if shapely.box(*tile_bounds(x, y, z)).intersects(city)]


class TileClient:
    def __init__(self, tok: str, session: requests.Session | None = None, timeout: int = 60):
        self.session = session or requests.Session()
        self.tok = tok
        self.timeout = timeout

    @retry(retry=retry_if_exception(_is_retryable), stop=stop_after_attempt(6),
           wait=wait_exponential(multiplier=2, max=60), reraise=True)
    def get(self, layer: str, x: int, y: int, z: int = ZOOM) -> bytes:
        url = TILE_URL.format(layer=layer, z=z, x=x, y=y)
        r = self.session.get(url, params={"access_token": self.tok}, timeout=self.timeout)
        if not r.ok:  # the token is in the request URL; keep it out of the error text
            raise requests.HTTPError(f"{r.status_code} for {url}", response=r)
        return r.content


def tile_path(kind: str, x: int, y: int):
    return TILE_DIR / kind / f"{ZOOM}_{x}_{y}.mvt"


def download(client: TileClient, tiles: list[tuple[int, int]], kinds=tuple(LAYERS), force: bool = False) -> int:

    todo = [(k, x, y) for k in kinds for x, y in tiles if force or not tile_path(k, x, y).exists()]
    log.info("mapillary: %d tiles to fetch (%d cached)", len(todo), len(kinds) * len(tiles) - len(todo))

    def fetch(job):
        k, x, y = job
        data = client.get(LAYERS[k], x, y)
        p = tile_path(k, x, y)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(p)

    with ThreadPoolExecutor(WORKERS) as pool:
        for i, _ in enumerate(pool.map(fetch, todo), 1):
            if i % 200 == 0:
                log.info("  %d / %d", i, len(todo))
    return len(todo)


# decoding
def decode_points(data: bytes, x: int, y: int, layer: str, z: int = ZOOM) -> pd.DataFrame:

    if not data:
        return pd.DataFrame()
    tiles = mvt.decode(data)
    if layer not in tiles:
        return pd.DataFrame()
    L = tiles[layer]
    ext = L.get("extent", 4096)
    rows = []
    for f in L["features"]:
        g = f["geometry"]
        if g["type"] != "Point":
            continue
        px, py = g["coordinates"]
        rows.append({**f["properties"], "px": px, "py": py})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    n = 2 ** z
    fx = x + df.pop("px") / ext
    fy = y + 1 - df.pop("py") / ext  # tile y runs up from the bottom edge
    df["lon"] = fx / n * 360 - 180
    df["lat"] = (fy.map(lambda t: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * t / n))))))
    return df


def build(tiles: list[tuple[int, int]]) -> tuple[pd.DataFrame, pd.DataFrame]:

    det, img = [], []
    for x, y in tiles:
        for kind, layer in (("points", "point"), ("signs", "traffic_sign")):
            d = decode_points(tile_path(kind, x, y).read_bytes(), x, y, layer)
            if len(d):
                det.append(d.assign(source=kind))
        d = decode_points(tile_path("coverage", x, y).read_bytes(), x, y, "image")
        if len(d):
            img.append(d)
    det = pd.concat(det, ignore_index=True).drop_duplicates(["source", "id"])
    det = det[["source", "id", "value", "first_seen_at", "last_seen_at", "lon", "lat"]]
    for c in ["first_seen_at", "last_seen_at"]:
        det[c] = pd.to_datetime(det[c], unit="ms")
    img = pd.concat(img, ignore_index=True).drop_duplicates("id")
    img = img[["id", "captured_at", "is_pano", "lon", "lat"]]
    img["captured_at"] = pd.to_datetime(img["captured_at"], unit="ms")
    return det.reset_index(drop=True), img.reset_index(drop=True)


def checks(det: pd.DataFrame, img: pd.DataFrame, n_tiles: int) -> list[str]:
    fails = []
    for kind in LAYERS:
        have = len(list((TILE_DIR / kind).glob(f"{ZOOM}_*.mvt")))
        if have < n_tiles:
            fails.append(f"{kind}: {have} of {n_tiles} tiles on disk")
    b = config.CITY_BBOX
    for name, df in (("detections", det), ("images", img)):
        inside = df["lat"].between(b["lat_min"] - 0.05, b["lat_max"] + 0.05) & \
                 df["lon"].between(b["lon_min"] - 0.05, b["lon_max"] + 0.05)
        if not inside.all():
            fails.append(f"{name}: {(~inside).sum()} points far outside the city")
        if df["id"].duplicated().any() and name == "images":
            fails.append("duplicate image ids")
    if not det["value"].str.startswith("object--traffic-light").any():
        fails.append("no traffic lights decoded")
    return fails


def run(force: bool = False) -> None:
    tiles = city_tiles()
    log.info("mapillary: %d tiles at zoom %d cover the city", len(tiles), ZOOM)
    download(TileClient(token()), tiles, force=force)
    det, img = build(tiles)
    MLY_DIR.mkdir(parents=True, exist_ok=True)
    det.to_parquet(DETECTIONS_PATH, index=False)
    img.to_parquet(IMAGES_PATH, index=False)
    log.info("detections: %d (%d kinds); images: %d", len(det), det["value"].nunique(), len(img))
    fails = checks(det, img, len(tiles))
    if fails:
        raise RuntimeError("mapillary checks failed: " + "; ".join(fails))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run()
