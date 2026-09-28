"""High-resolution basemap mosaics (Google satellite, Bing/Azure Maps aerial)
for a study area, written as georeferenced GeoTIFFs (EPSG:3857).

These basemaps are undated composites (typically the newest cloud-free scene
per area, anywhere from 1 to 5 years old) so they are used for *visual QA
and digitising the current extent* - the multi-year analysis comes from
Landsat/Sentinel-2. Check the provider's terms of use for storing tiles.

Providers
---------
google  Maps Static API, maptype=satellite. One request per XYZ tile
        (size 256x256, scale 2 -> 512 px). Needs GOOGLE_MAPS_API_KEY.
azure   Azure Maps Render "map/tile", tilesetId=microsoft.imagery (the
        successor to Bing Maps aerial imagery). Needs AZURE_MAPS_KEY.
bing    Legacy Bing Maps aerial via the Imagery Metadata API; only works if
        your Bing Maps for Enterprise licence is still active. BING_MAPS_KEY.
"""
from __future__ import annotations

import io
import logging
import math
import time
from pathlib import Path

import mercantile
import numpy as np
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_bounds

log = logging.getLogger("wetscan.tiles")
TILE = 256
UA = {"User-Agent": "wetscan/0.1 (wetland screening; contact: solasenergy.com)"}


def _quadkey(x: int, y: int, z: int) -> str:
    q = ""
    for i in range(z, 0, -1):
        d = 0
        mask = 1 << (i - 1)
        if x & mask:
            d += 1
        if y & mask:
            d += 2
        q += str(d)
    return q


class Provider:
    name = "base"
    px = TILE

    def __init__(self, key: str):
        self.key = key
        self.session = requests.Session()
        self.session.headers.update(UA)

    def fetch(self, x: int, y: int, z: int) -> Image.Image:
        raise NotImplementedError


class Google(Provider):
    name = "google"
    px = TILE * 2  # scale=2

    def fetch(self, x, y, z):
        b = mercantile.bounds(x, y, z)
        lat, lon = (b.north + b.south) / 2, (b.east + b.west) / 2
        # Static Maps centres on the mercator midpoint, which is not the
        # arithmetic mean of the latitudes - compute it properly.
        ymid = (mercantile.xy(0, b.north)[1] + mercantile.xy(0, b.south)[1]) / 2
        lat = math.degrees(math.atan(math.sinh(ymid / 6378137.0)))
        r = self.session.get("https://maps.googleapis.com/maps/api/staticmap", params={
            "center": f"{lat:.7f},{lon:.7f}", "zoom": z, "size": f"{TILE}x{TILE}", "scale": 2,
            "maptype": "satellite", "format": "png", "key": self.key}, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")


class Azure(Provider):
    name = "azure"

    def fetch(self, x, y, z):
        r = self.session.get("https://atlas.microsoft.com/map/tile", params={
            "api-version": "2024-04-01", "tilesetId": "microsoft.imagery", "zoom": z, "x": x, "y": y,
            "tileSize": "256", "subscription-key": self.key}, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")


class Bing(Provider):
    name = "bing"

    def __init__(self, key):
        super().__init__(key)
        meta = self.session.get("https://dev.virtualearth.net/REST/v1/Imagery/Metadata/Aerial",
                                params={"key": key, "output": "json"}, timeout=60).json()
        res = meta["resourceSets"][0]["resources"][0]
        self.template = res["imageUrl"]
        self.subdomains = res.get("imageUrlSubdomains", ["t0"])

    def fetch(self, x, y, z):
        url = (self.template.replace("{subdomain}", self.subdomains[(x + y) % len(self.subdomains)])
               .replace("{quadkey}", _quadkey(x, y, z)).replace("{culture}", "en-US"))
        r = self.session.get(url, timeout=60)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")


def make_provider(name: str, keys) -> Provider:
    if name == "google":
        if not keys.google_maps:
            raise RuntimeError("GOOGLE_MAPS_API_KEY not set")
        return Google(keys.google_maps)
    if name == "azure":
        if not keys.azure_maps:
            raise RuntimeError("AZURE_MAPS_KEY not set")
        return Azure(keys.azure_maps)
    if name == "bing":
        if not keys.bing_maps:
            raise RuntimeError("BING_MAPS_KEY not set")
        return Bing(keys.bing_maps)
    raise ValueError(name)


def mosaic(provider: Provider, bounds_wgs84, zoom: int, out: Path, max_tiles: int = 400,
           sleep_s: float = 0.05) -> Path:
    """Fetch every XYZ tile covering ``bounds_wgs84`` and write a GeoTIFF (EPSG:3857)."""
    w, s, e, n = bounds_wgs84
    tiles = list(mercantile.tiles(w, s, e, n, zoom))
    if len(tiles) > max_tiles:
        raise RuntimeError(f"{len(tiles)} tiles at z{zoom} exceeds max_tiles={max_tiles}; lower the zoom")
    xs = [t.x for t in tiles]; ys = [t.y for t in tiles]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    px = provider.px
    canvas = np.zeros(((y1 - y0 + 1) * px, (x1 - x0 + 1) * px, 3), dtype=np.uint8)
    for i, t in enumerate(tiles):
        for attempt in range(3):
            try:
                img = provider.fetch(t.x, t.y, t.z)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 2:
                    log.warning("tile %s failed: %s", t, exc)
                    img = Image.new("RGB", (px, px))
                time.sleep(1 + attempt)
        if img.size != (px, px):
            img = img.resize((px, px))
        r, c = (t.y - y0) * px, (t.x - x0) * px
        canvas[r:r + px, c:c + px] = np.asarray(img)
        time.sleep(sleep_s)
        if i % 50 == 0:
            log.info("%s: %d/%d tiles", provider.name, i + 1, len(tiles))
    ul = mercantile.xy_bounds(x0, y0, zoom)
    lr = mercantile.xy_bounds(x1, y1, zoom)
    tf = from_bounds(ul.left, lr.bottom, lr.right, ul.top, canvas.shape[1], canvas.shape[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", driver="GTiff", height=canvas.shape[0], width=canvas.shape[1], count=3,
                       dtype="uint8", crs="EPSG:3857", transform=tf, compress="jpeg", photometric="ycbcr",
                       tiled=True) as dst:
        dst.write(np.moveaxis(canvas, -1, 0))
        dst.update_tags(provider=provider.name, zoom=str(zoom), fetched=time.strftime("%Y-%m-%d"))
    return out


def zoom_for_resolution(lat: float, target_m_per_px: float = 0.6, px: int = TILE) -> int:
    """Smallest zoom whose ground resolution at ``lat`` is <= target."""
    for z in range(10, 21):
        res = 156543.03 * math.cos(math.radians(lat)) / (2 ** z) * (TILE / px)
        if res <= target_m_per_px:
            return z
    return 20
