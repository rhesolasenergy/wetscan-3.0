"""Per-year surface-water and wet-vegetation masks from the full Landsat and
Sentinel-2 archive (Google Earth Engine), downloaded as local GeoTIFFs.

Sensors and years
-----------------
  1972-1983  Landsat 1-3 MSS (LM01/02/03 C02 T1, 60 m, raw DN, no SWIR)
  1982-2012  Landsat 4/5 TM  (LT04/LT05 C02 T1_L2 surface reflectance, 30 m)
  1999-2022  Landsat 7 ETM+  (LE07, SLC-off gaps after 2003 filled by compositing)
  2013-      Landsat 8/9 OLI (LC08/LC09, 30 m)
  2017-      Sentinel-2 MSI  (S2_SR_HARMONIZED, 10 m; 2015-16 L1C via S2_HARMONIZED)

Every image is harmonised to the bands  blue, green, red, nir, swir1  (swir1
missing for MSS) with reflectance in 0-1, cloud/shadow/snow masked from the
QA bands, and the following indices computed:

  MNDWI = (green - swir1) / (green + swir1)      open water (Xu 2006)
  NDWI  = (green - nir)   / (green + nir)        open water, MSS fallback (McFeeters 1996)
  NDVI  = (nir - red)     / (nir + red)
  NDMI  = (nir - swir1)   / (nir + swir1)        canopy / soil moisture (Gao 1996)

For each year two seasonal windows are composited (see config.Season):

  spring : 75th percentile of the water index  -> w_spr   (maximum inundation)
  late   : median water index                  -> w_late  (persistent water)
           median NDMI & NDVI                  -> wv_late (wet vegetation)

Per-year GeoTIFF bands (uint8, 255 = no observation):
  1 w_spr, 2 w_late, 3 wv_late, 4 n_spr (obs count), 5 n_late (obs count)

A single "static" GeoTIFF is also written with ESA WorldCover 2021 land cover,
JRC Global Surface Water occurrence (1984-2021), Copernicus GLO-30 slope and
tree-cover fraction, which the classifier uses as priors.
"""
from __future__ import annotations

import io
import json
import logging
import math
import zipfile
from pathlib import Path

import numpy as np
import requests

from .config import Config

log = logging.getLogger("wetscan.imagery")

MSS_LAST_YEAR = 1983
S2_FIRST_YEAR = 2017


# --------------------------------------------------------------------------- #
# Earth Engine session
# --------------------------------------------------------------------------- #
def init_ee(project: str | None) -> None:
    import ee
    try:
        ee.Initialize(project=project) if project else ee.Initialize()
    except Exception:  # first run on a machine -> interactive auth
        ee.Authenticate()
        ee.Initialize(project=project) if project else ee.Initialize()


# --------------------------------------------------------------------------- #
# Sensor harmonisation
# --------------------------------------------------------------------------- #
def _landsat_sr(img, bands: dict):
    """Landsat C2 L2: scale reflectance and mask with QA_PIXEL."""
    import ee
    qa = img.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 1).Or(qa.bitwiseAnd(1 << 3)).Or(qa.bitwiseAnd(1 << 4)).Or(qa.bitwiseAnd(1 << 5))
    sr = img.select(list(bands.values()), list(bands.keys())).multiply(0.0000275).add(-0.2)
    sr = sr.updateMask(cloud.Not()).updateMask(sr.select("blue").gt(0)).updateMask(sr.select("nir").lt(1))
    return sr.toFloat().copyProperties(img, ["system:time_start"])


def _mss(img, bands: dict):
    """Landsat MSS C2 T1: raw DN. No SWIR. Crude cloud screen from QA_PIXEL
    plus a brightness test."""
    import ee
    qa = img.select("QA_PIXEL")
    cloud = qa.bitwiseAnd(1 << 3).Or(qa.bitwiseAnd(1 << 4))
    dn = img.select(list(bands.values()), list(bands.keys())).toFloat().divide(255.0)
    bright = dn.select("green").add(dn.select("red")).gt(1.2)
    dn = dn.updateMask(cloud.Not()).updateMask(bright.Not())
    return dn.copyProperties(img, ["system:time_start"])


def _s2(img):
    import ee
    scl = img.select("SCL")
    bad = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10)).Or(scl.eq(11)).Or(scl.eq(1))
    sr = img.select(["B2", "B3", "B4", "B8", "B11"], ["blue", "green", "red", "nir", "swir1"]).multiply(1e-4)
    return sr.updateMask(bad.Not()).toFloat().copyProperties(img, ["system:time_start"])


def harmonised_collection(region, start: str, end: str, year: int, max_cloud: float):
    """Merged, harmonised ImageCollection for one date window."""
    import ee
    cols = []
    if year <= MSS_LAST_YEAR:
        b123 = {"green": "B4", "red": "B5", "nir": "B7"}
        for cid in ("LANDSAT/LM01/C02/T1", "LANDSAT/LM02/C02/T1", "LANDSAT/LM03/C02/T1"):
            cols.append(ee.ImageCollection(cid).filterBounds(region).filterDate(start, end)
                        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
                        .map(lambda i, b=b123: _mss(i, b)))
        # Landsat 4/5 MSS bands are numbered 1-4
        b45 = {"green": "B1", "red": "B2", "nir": "B4"}
        for cid in ("LANDSAT/LM04/C02/T1", "LANDSAT/LM05/C02/T1"):
            cols.append(ee.ImageCollection(cid).filterBounds(region).filterDate(start, end)
                        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
                        .map(lambda i, b=b45: _mss(i, b)))
    else:
        tm = {"blue": "SR_B1", "green": "SR_B2", "red": "SR_B3", "nir": "SR_B4", "swir1": "SR_B5"}
        oli = {"blue": "SR_B2", "green": "SR_B3", "red": "SR_B4", "nir": "SR_B5", "swir1": "SR_B6"}
        for cid, b in (("LANDSAT/LT04/C02/T1_L2", tm), ("LANDSAT/LT05/C02/T1_L2", tm),
                       ("LANDSAT/LE07/C02/T1_L2", tm), ("LANDSAT/LC08/C02/T1_L2", oli),
                       ("LANDSAT/LC09/C02/T1_L2", oli)):
            cols.append(ee.ImageCollection(cid).filterBounds(region).filterDate(start, end)
                        .filter(ee.Filter.lt("CLOUD_COVER", max_cloud))
                        .map(lambda i, b=b: _landsat_sr(i, b)))
        if year >= S2_FIRST_YEAR:
            cols.append(ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
                        .filterDate(start, end)
                        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", max_cloud)).map(_s2))
    merged = cols[0]
    for c in cols[1:]:
        merged = merged.merge(c)
    return merged


def add_indices(img, has_swir: bool):
    ndvi = img.normalizedDifference(["nir", "red"]).rename("ndvi")
    ndwi = img.normalizedDifference(["green", "nir"]).rename("ndwi")
    out = img.addBands([ndvi, ndwi])
    if has_swir:
        out = out.addBands(img.normalizedDifference(["green", "swir1"]).rename("mndwi"))
        out = out.addBands(img.normalizedDifference(["nir", "swir1"]).rename("ndmi"))
    return out


# --------------------------------------------------------------------------- #
# Per-year mask image
# --------------------------------------------------------------------------- #
def year_masks(region, year: int, cfg: Config):
    """Return an ee.Image with bands w_spr, w_late, wv_late, n_spr, n_late."""
    import ee
    """Return an ee.Image of per-year *index composites* (not masks), so that
    all thresholding happens locally and can be re-tuned without re-downloading.

    Bands (int16, index * 10000; NODATA = -32768):
      1 wi_spr_p75   spring 75th-percentile water index (MNDWI; NDWI for MSS)
      2 wi_late      late-summer median water index
      3 ndmi_late    late-summer median NDMI  (NDWI-based proxy for MSS)
      4 ndvi_late    late-summer median NDVI
      5 n_spr        spring observation count
      6 n_late       late-summer observation count
    """
    import ee
    s, t = cfg.season, cfg.thresholds
    has_swir = year > MSS_LAST_YEAR
    windex = "mndwi" if has_swir else "ndwi"
    mindex = "ndmi" if has_swir else "ndwi"

    spring = harmonised_collection(region, f"{year}-{s.spring_start}", f"{year}-{s.spring_end}", year, t.max_cloud_pct)
    late = harmonised_collection(region, f"{year}-{s.late_start}", f"{year}-{s.late_end}", year, t.max_cloud_pct)
    spring = spring.map(lambda i: add_indices(i, has_swir))
    late = late.map(lambda i: add_indices(i, has_swir))

    n_spr = spring.select(windex).count().rename("n_spr")
    n_late = late.select(windex).count().rename("n_late")
    wi_spr = spring.select(windex).reduce(ee.Reducer.percentile([75])).rename("wi_spr_p75")
    late_med = late.median()
    wi_late = late_med.select(windex).rename("wi_late")
    ndmi_late = late_med.select(mindex).rename("ndmi_late")
    ndvi_late = late_med.select("ndvi").rename("ndvi_late")

    idx = ee.Image.cat([wi_spr, wi_late, ndmi_late, ndvi_late]).multiply(10000).round()
    img = (ee.Image.cat([idx, n_spr, n_late]).toInt16().unmask(NODATA_I16).clip(region))
    return img.set({"year": year, "sensor": "MSS" if not has_swir else "TM/ETM/OLI/S2"})


NODATA_I16 = -32768
YEAR_BANDS = ["wi_spr_p75", "wi_late", "ndmi_late", "ndvi_late", "n_spr", "n_late"]
STACK_VERSION = 2


def static_layers(region):
    """Land-cover and terrain priors for classification."""
    import ee
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover")
    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).rename("jrc_occurrence")
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic()
    slope = ee.Terrain.slope(dem).rename("slope_deg")
    tree = ee.Image("UMD/hansen/global_forest_change_2023_v1_11").select("treecover2000").rename("treecover_pct")
    return ee.Image.cat([wc, jrc, slope, tree]).toFloat().clip(region)


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def _download_geotiff(img, region, scale: int, crs: str, out: Path, bands: list[str]) -> Path:
    """getDownloadURL -> GeoTIFF on disk. Tiles the request when EE rejects it
    for size and mosaics the pieces."""
    import ee
    import rasterio
    from rasterio.merge import merge

    def _one(reg, dest: Path):
        url = img.select(bands).getDownloadURL({
            "region": reg, "scale": scale, "crs": crs,
            "format": "GEO_TIFF", "filePerBand": False})
        r = requests.get(url, timeout=600)

        if not r.ok:
            raise RuntimeError(
                f"Earth Engine download failed ({r.status_code}): {r.text[:3000]}"
            )

        r.raise_for_status()
        if r.headers.get("content-type", "").startswith("application/zip"):
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                name = [n for n in z.namelist() if n.endswith(".tif")][0]
                dest.write_bytes(z.read(name))
        else:
            dest.write_bytes(r.content)
        return dest
    try:
        return _one(region, out)
    except Exception as e:  # noqa: BLE001
        if "size" not in str(e).lower() and "too large" not in str(e).lower() and "limit" not in str(e).lower():
            raise
        log.info("request too large - tiling %s", out.name)

    # tile the bounding box into a grid and mosaic
    b = region.bounds().getInfo()["coordinates"][0]
    xs = [p[0] for p in b]; ys = [p[1] for p in b]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    n = 2
    while True:
        parts = []
        try:
            dx, dy = (maxx - minx) / n, (maxy - miny) / n
            for i in range(n):
                for j in range(n):
                    reg = ee.Geometry.Rectangle([minx + i * dx, miny + j * dy, minx + (i + 1) * dx, miny + (j + 1) * dy])
                    parts.append(_one(reg, out.with_name(f"{out.stem}_t{i}{j}.tif")))
            break
        except Exception as e:  # noqa: BLE001
            for p in parts:
                p.unlink(missing_ok=True)
            n *= 2
            if n > 16:
                raise RuntimeError(f"AOI too large to download even in {n}x{n} tiles: {e}") from e
    srcs = [rasterio.open(p) for p in parts]
    mosaic, tf = merge(srcs, nodata=srcs[0].nodata)
    meta = srcs[0].meta.copy()
    meta.update(height=mosaic.shape[1], width=mosaic.shape[2], transform=tf)
    for s in srcs:
        s.close()
    with rasterio.open(out, "w", **meta) as dst:
        dst.write(mosaic)
    for p in parts:
        p.unlink(missing_ok=True)
    return out


def fetch_year_stack(bounds_wgs84: tuple[float, float, float, float], crs: str, cfg: Config,
                     out_dir: Path, years: list[int], overwrite: bool = False) -> dict[int, Path]:
    """Download one GeoTIFF per year (cached) plus static layers. Returns {year: path}."""
    import ee
    out_dir.mkdir(parents=True, exist_ok=True)
    region = ee.Geometry.Rectangle(list(bounds_wgs84))
    paths: dict[int, Path] = {}

    static = out_dir / "static_layers.tif"
    if overwrite or not static.exists():
        log.info("downloading static layers")
        _download_geotiff(static_layers(region), region, cfg.scale_m, crs, static,
                          ["worldcover", "jrc_occurrence", "slope_deg", "treecover_pct"])

    failures: dict[str, str] = {}
    for y in years:
        p = out_dir / f"indices_{y}.tif"
        if p.exists() and not overwrite:
            paths[y] = p
            continue
        log.info("downloading index composites for %d", y)
        try:
            _download_geotiff(year_masks(region, y, cfg), region, cfg.scale_m, crs, p, YEAR_BANDS)
            paths[y] = p
        except Exception as e:  # noqa: BLE001
            # an empty collection (no cloud-free scenes) raises inside EE; record and move on
            msg = f"{type(e).__name__}: {e}"
            failures[str(y)] = msg
            log.warning("year %d skipped: %s", y, msg[:300])
    (out_dir / "years.json").write_text(json.dumps({str(k): str(v) for k, v in paths.items()}, indent=2))
    (out_dir / "years_failed.json").write_text(json.dumps(failures, indent=2))
    (out_dir / "stack_version.txt").write_text(str(STACK_VERSION))
    if failures:
        log.warning("%d year(s) failed - see %s", len(failures), out_dir / "years_failed.json")
    return paths


def scene_inventory(bounds_wgs84, cfg: Config, years: list[int]) -> list[dict]:
    """How many cloud-screened observations exist per year and window (for the report)."""
    import ee
    region = ee.Geometry.Rectangle(list(bounds_wgs84))
    s, t = cfg.season, cfg.thresholds
    rows = []
    for y in years:
        spr = harmonised_collection(region, f"{y}-{s.spring_start}", f"{y}-{s.spring_end}", y, t.max_cloud_pct).size()
        lat = harmonised_collection(region, f"{y}-{s.late_start}", f"{y}-{s.late_end}", y, t.max_cloud_pct).size()
        rows.append({"year": y, "spring_scenes": spr, "late_scenes": lat})
    # one server round-trip
    vals = ee.List([[r["spring_scenes"], r["late_scenes"]] for r in rows]).getInfo()
    for r, (a, b) in zip(rows, vals):
        r["spring_scenes"], r["late_scenes"] = a, b
        r["sensor"] = "MSS" if y <= MSS_LAST_YEAR else "Landsat/S2"
    return rows
