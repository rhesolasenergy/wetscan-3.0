"""Build a synthetic site (v2 index composites + static layers + precipitation)
so the offline half of the pipeline can be exercised without Earth Engine.

The landscape deliberately contains the failure mode seen on the first Portage
run: large cropped fields that are lush (high NDMI/NDVI) in wet years. The
tests assert they are NOT mapped as wetland."""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from wetscan import precip
from wetscan.aoi import split_sites

YEARS = list(range(1972, 2026))
SHAPE = (300, 300)  # 10 m pixels -> 3 km x 3 km
CRS = "EPSG:32612"
ORIGIN = (400000.0, 5700000.0)
NODATA = -32768


def synthetic_precip(seed=1) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.period_range("1970-01", "2025-09", freq="M")
    base = np.array([18, 12, 18, 25, 45, 75, 60, 50, 35, 20, 15, 16])  # Alberta-ish monthly mm
    vals = []
    yr_factor = {}
    for p in idx:
        wy = p.year + 1 if p.month >= 10 else p.year
        yr_factor.setdefault(wy, rng.lognormal(0, 0.28))
        vals.append(max(0, base[p.month - 1] * yr_factor[wy] * rng.lognormal(0, 0.25)))
    return pd.Series(vals, index=idx, name="precip_mm")


def _disk(cy, cx, r):
    yy, xx = np.ogrid[:SHAPE[0], :SHAPE[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2


def build(out: Path, year_class: pd.Series) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    mdir = out / "masks"
    mdir.mkdir(exist_ok=True)
    tf = from_origin(ORIGIN[0], ORIGIN[1], 10, 10)
    rng = np.random.default_rng(7)

    # wetland templates (rows, cols):
    pond = _disk(80, 80, 21)            # permanent shallow open water
    pond_max = _disk(80, 80, 26)
    marsh_core = _disk(200, 90, 10)     # seasonal marsh: spring water, dries by late summer
    marsh_max = _disk(200, 90, 22)
    peat = np.zeros(SHAPE, bool); peat[40:110, 180:270] = True   # treed fen: saturated, no open water
    ephem = _disk(240, 230, 9)          # ephemeral depression, wet only in very wet springs
    field = np.zeros(SHAPE, bool); field[150:290, 150:290] = True  # lush cropland (the trap)
    pasture = np.zeros(SHAPE, bool); pasture[0:40, 0:300] = True   # grassland strip, uniformly moist

    wc = np.full(SHAPE, 30, np.uint8)   # grassland
    wc[field] = 40                      # cropland
    wc[peat] = 10                       # tree cover
    wc[pond] = 80
    wc[marsh_max & ~marsh_core] = 90
    tree = np.where(peat, 70.0, 2.0)
    jrc = np.where(pond, 95.0, np.where(marsh_core, 30.0, 0.0))
    slope = np.full(SHAPE, 0.5, np.float32)

    paths = {}
    for y in YEARS:
        cls = year_class.get(y, "normal")
        wet_f = {"very_dry": 0.0, "dry": 0.35, "normal": 0.65, "wet": 0.9, "very_wet": 1.0}[cls]
        mss = y <= 1983
        # --- water index composites ---
        w_spr_mask = (pond_max if wet_f >= 0.9 else pond) | (marsh_max if wet_f >= 0.6 else (marsh_core if wet_f > 0.2 else np.zeros(SHAPE, bool)))
        if wet_f >= 0.9:
            w_spr_mask |= ephem
        w_late_mask = pond | (marsh_core if wet_f >= 0.9 else np.zeros(SHAPE, bool))
        wi_spr = np.where(w_spr_mask, 0.35, -0.45) + rng.normal(0, 0.03, SHAPE)
        wi_late = np.where(w_late_mask, 0.30, -0.50) + rng.normal(0, 0.03, SHAPE)
        # --- late-summer moisture / greenness ---
        # background dry grassland: NDMI 0.05, NDVI 0.35
        ndmi = np.full(SHAPE, 0.05) + rng.normal(0, 0.02, SHAPE)
        ndvi = np.full(SHAPE, 0.35) + rng.normal(0, 0.03, SHAPE)
        # cropland: green and moist in wet years (NDMI 0.35-0.45!), stubble in dry years
        ndmi[field] = 0.10 + 0.35 * wet_f + rng.normal(0, 0.02, field.sum())
        ndvi[field] = 0.30 + 0.45 * wet_f + rng.normal(0, 0.03, field.sum())
        # pasture strip: uniformly moist (0.32) - anomaly is zero so must not qualify
        ndmi[pasture] = 0.32 + rng.normal(0, 0.02, pasture.sum())
        ndvi[pasture] = 0.55
        # marsh fringe + peatland: wetter than surroundings every year
        fringe = marsh_max & ~w_late_mask
        ndmi[fringe] = 0.42 + rng.normal(0, 0.02, fringe.sum()); ndvi[fringe] = 0.6
        ndmi[peat] = 0.45 + rng.normal(0, 0.02, peat.sum()); ndvi[peat] = 0.55
        # a one-off lush patch in a single year (should not persist)
        if y == 2005:
            oneoff = _disk(30, 250, 8); ndmi[oneoff] = 0.5; ndvi[oneoff] = 0.6
        n_spr = np.full(SHAPE, 1 if mss else 3, np.int16)
        n_late = np.full(SHAPE, 1 if mss else 4, np.int16)
        if y in (1975, 1979, 1988):  # cloudy years with no data
            n_spr[:] = 0; n_late[:] = 0
        bands = np.stack([wi_spr, wi_late, ndmi, ndvi])
        arr = np.rint(bands * 10000).astype(np.int16)
        arr[:, n_spr == 0] = NODATA
        arr = np.concatenate([arr, n_spr[None], n_late[None]]).astype(np.int16)
        p = mdir / f"indices_{y}.tif"
        with rasterio.open(p, "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=6, dtype="int16",
                           crs=CRS, transform=tf, nodata=NODATA) as dst:
            dst.write(arr)
            dst.descriptions = ("wi_spr_p75", "wi_late", "ndmi_late", "ndvi_late", "n_spr", "n_late")
        paths[y] = p
    with rasterio.open(mdir / "static_layers.tif", "w", driver="GTiff", height=SHAPE[0], width=SHAPE[1], count=4,
                       dtype="float32", crs=CRS, transform=tf) as dst:
        dst.write(np.stack([wc.astype(np.float32), jrc.astype(np.float32), slope, tree.astype(np.float32)]))
        dst.descriptions = ("worldcover", "jrc_occurrence", "slope_deg", "treecover_pct")
    (mdir / "years.json").write_text(json.dumps({str(k): str(v) for k, v in paths.items()}))
    (mdir / "years_failed.json").write_text(json.dumps({"1973": "EEException: Image.select: no bands (no cloud-free scenes)"}))
    return {"pond": pond, "marsh": marsh_max, "peat": peat, "ephem": ephem, "field": field, "pasture": pasture}


def synthetic_site() -> dict:
    # project area = central 1.2 km square; study area = +2 km
    x0, y0 = ORIGIN[0] + 900, ORIGIN[1] - 2100
    parcel = box(x0, y0, x0 + 1200, y0 + 1200)
    g = gpd.GeoDataFrame({"name": ["NW-1-1-1-W4M", "800m Notification"], "folder": ["Synthetic (Site 1)"] * 2},
                         geometry=[parcel, parcel.buffer(800)], crs=CRS).to_crs("EPSG:4326")
    return split_sites(g, 2000.0)[0]
