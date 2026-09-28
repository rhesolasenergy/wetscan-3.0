"""Greatest wetland extent and per-year statistics from the stack of yearly masks.

Inputs are the per-year GeoTIFFs written by imagery.fetch_year_stack (or any
rasters with the same band layout). Everything here is plain numpy / rasterio
so it runs offline and is unit-testable.

Definitions
-----------
wet(y)          = w_spr(y) | w_late(y) | wv_late(y)        (MSS years: water only,
                                                            unless include_mss_veg)
observed(y)     = pixel had >= 1 cloud-free observation in either window
greatest extent = pixels with wet(y) in >= min_years years, after removing
                  slivers, filling holes and light smoothing
f_spring        = (# years w_spr) / (# years observed in spring)
f_late          = (# years w_late) / (# years observed late)
f_wet           = (# years wet)    / (# years observed)
f_late_wet_yrs  = f_late restricted to wet / very_wet precipitation years
f_late_dry_yrs  = f_late restricted to dry / very_dry precipitation years
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from scipy import ndimage
from shapely.geometry import shape
from shapely.ops import unary_union

from .config import Thresholds

log = logging.getLogger("wetscan.extent")
NODATA = 255


@dataclass
class YearStack:
    years: list[int]
    w_spr: np.ndarray    # (n_years, rows, cols) bool
    w_late: np.ndarray
    wv_late: np.ndarray
    obs_spr: np.ndarray  # bool: had observations
    obs_late: np.ndarray
    transform: rasterio.Affine
    crs: str
    is_mss: np.ndarray   # (n_years,) bool
    pixel_area_m2: float
    static: dict[str, np.ndarray] = field(default_factory=dict)
    moist_late: np.ndarray | None = None  # absolute NDMI test (peatland exception)


NODATA_I16 = -32768
WC_CROP, WC_BUILT, WC_WATER, WC_HERB_WET, WC_TREE, WC_SHRUB = 40, 50, 80, 90, 10, 20


def _read_static(static_path: Path | None) -> dict[str, np.ndarray]:
    static = {}
    if static_path and Path(static_path).exists():
        with rasterio.open(static_path) as src:
            names = list(src.descriptions or [])
            if not names or not any(names):
                names = ["worldcover", "jrc_occurrence", "slope_deg", "treecover_pct"][: src.count]
            for i, name in enumerate(names):
                static[name or f"band{i+1}"] = src.read(i + 1)
    return static


def _local_mean(a: np.ndarray, radius_px: int) -> np.ndarray:
    """Mean of ``a`` over a (2r+1)^2 window ignoring NaN."""
    valid = ~np.isnan(a)
    size = 2 * radius_px + 1
    s = ndimage.uniform_filter(np.where(valid, a, 0.0), size=size, mode="nearest")
    n = ndimage.uniform_filter(valid.astype(np.float64), size=size, mode="nearest")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 1e-6, s / n, np.nan)


def load_stack(year_paths: dict[int, Path], static_path: Path | None = None,
               thr: Thresholds = Thresholds(), mss_last_year: int = 1983,
               peat_region: bool = False) -> YearStack:
    """Read per-year index composites and derive the wet masks locally.

    Rules (all tunable in config.Thresholds):

    open water, spring  : wi_spr_p75 > mndwi_water          (ndwi_water_mss for MSS)
    open water, late    : wi_late    > mndwi_water
    wet vegetation      : the pixel is *wetter than its surroundings* - late NDMI
                          minus the mean NDMI within wetveg_window_m exceeds
                          ndmi_anomaly - AND ndmi_late > ndmi_wet AND ndvi_late >
                          ndvi_veg_min, AND not open water. Cropland / built-up
                          (ESA WorldCover) and slopes > max_slope_deg never qualify.
                          MSS years (no SWIR) never contribute wet vegetation.

    A further landscape test (persistence and proximity to water) is applied
    in ``support_mask`` once all years are loaded, because it needs the whole
    record.
    """
    years = sorted(year_paths)
    static = _read_static(static_path)
    w_spr, w_late, wv, o_spr, o_late, moist = [], [], [], [], [], []
    ref = None
    for y in years:
        with rasterio.open(year_paths[y]) as src:
            if ref is None:
                ref = (src.transform, src.crs.to_string(), src.shape)
            a = src.read()
            desc = list(src.descriptions or [])
        if a.shape[1:] != ref[2]:
            raise ValueError(f"raster for {y} has a different shape {a.shape[1:]} vs {ref[2]}")
        if a.shape[0] != 6 or a.dtype != np.int16:
            raise ValueError(
                f"{year_paths[y]} is not a v2 index composite (6 int16 bands). Delete the masks folder "
                "or run with --overwrite to re-download with the current version.")
        wi_spr = np.where(a[0] == NODATA_I16, np.nan, a[0] / 10000.0)
        wi_late = np.where(a[1] == NODATA_I16, np.nan, a[1] / 10000.0)
        ndmi = np.where(a[2] == NODATA_I16, np.nan, a[2] / 10000.0)
        ndvi = np.where(a[3] == NODATA_I16, np.nan, a[3] / 10000.0)
        os_ = (a[4] != NODATA_I16) & (a[4] > 0) & ~np.isnan(wi_spr)
        ol_ = (a[5] != NODATA_I16) & (a[5] > 0) & ~np.isnan(wi_late)
        is_mss = y <= mss_last_year
        wthr = thr.ndwi_water_mss if is_mss else thr.mndwi_water
        ws = os_ & (wi_spr > wthr)
        wl = ol_ & (wi_late > wthr)

        if is_mss:
            wv_y = np.zeros(ref[2], bool)
            mo_y = np.zeros(ref[2], bool)
        else:
            px = abs(ref[0].a)
            r = max(1, int(round(thr.wetveg_window_m / px)))
            anomaly = ndmi - _local_mean(ndmi, r)
            wv_y = (ol_ & ~wl & (anomaly > thr.ndmi_anomaly) & (ndmi > thr.ndmi_wet) & (ndvi > thr.ndvi_veg_min))
            # absolute test used only by the peatland exception: large fens/bogs have
            # no "surroundings" inside the window, so the anomaly goes to zero there
            mo_y = ol_ & ~wl & (ndmi > thr.ndmi_peat) & (ndvi > thr.ndvi_veg_min)
        w_spr.append(ws); w_late.append(wl); wv.append(wv_y); o_spr.append(os_); o_late.append(ol_); moist.append(mo_y)
    tf, crs, shp = ref

    # static exclusions for the wet-vegetation rule
    excl = np.zeros(shp, bool)
    if "worldcover" in static:
        excl |= np.isin(static["worldcover"], [WC_CROP, WC_BUILT])
    if "slope_deg" in static:
        excl |= static["slope_deg"] > thr.max_slope_deg
    wv = np.stack(wv)
    wv[:, excl] = False
    moist = np.stack(moist)
    moist[:, excl] = False

    st = YearStack(
        years=years, w_spr=np.stack(w_spr), w_late=np.stack(w_late), wv_late=wv,
        obs_spr=np.stack(o_spr), obs_late=np.stack(o_late), transform=tf, crs=crs,
        is_mss=np.array([y <= mss_last_year for y in years]),
        pixel_area_m2=abs(tf.a * tf.e), static=static, moist_late=moist,
    )
    keep = support_mask(st, thr, peat_region)
    st.wv_late = (st.wv_late | (moist if peat_region else False)) & keep
    return st


def support_mask(st: YearStack, thr: Thresholds, peat_region: bool) -> np.ndarray:
    """Landscape test for wet-vegetation pixels, applied across the whole record:

    * persistence - wetter-than-surroundings in >= wetveg_min_persistence of the
      observed late-summer years (crop rotation and hay cuts are not persistent);
    * hydrological support - within water_proximity_m of a pixel that held open
      water in any year, or mapped as water / herbaceous wetland by WorldCover,
      or JRC surface-water occurrence > 0;
    * peatland exception (boreal / foothills only) - persistently anomalous
      pixels under tree / shrub / herbaceous-wetland cover are kept even without
      nearby open water, because fens and bogs rarely show open water.
    """
    n_late = st.obs_late.sum(0)
    f_wv = _safe_div(st.wv_late.sum(0), n_late)
    persistent = np.nan_to_num(f_wv) >= thr.wetveg_min_persistence

    ever_water = (st.w_spr | st.w_late).any(0)
    wc = st.static.get("worldcover")
    if wc is not None:
        ever_water |= np.isin(wc, [WC_WATER, WC_HERB_WET])
    if "jrc_occurrence" in st.static:
        ever_water |= st.static["jrc_occurrence"] > 0
    px = np.sqrt(st.pixel_area_m2)
    r = max(1, int(round(thr.water_proximity_m / px)))
    near_water = ndimage.binary_dilation(ever_water, structure=np.ones((3, 3), bool), iterations=r)

    keep = persistent & near_water
    if peat_region and wc is not None and st.moist_late is not None:
        peat_cover = np.isin(wc, [WC_TREE, WC_SHRUB, WC_HERB_WET])
        f_moist = np.nan_to_num(_safe_div(st.moist_late.sum(0), n_late))
        keep |= (f_moist >= thr.wetveg_min_persistence) & peat_cover
    return keep


# --------------------------------------------------------------------------- #
# Frequencies and greatest extent
# --------------------------------------------------------------------------- #
def _safe_div(n, d):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0, n / np.maximum(d, 1), np.nan)


def wet_any(st: YearStack, include_mss_veg: bool = False) -> np.ndarray:
    veg = st.wv_late.copy()
    if not include_mss_veg:
        veg[st.is_mss] = False
    return st.w_spr | st.w_late | veg


def frequency_layers(st: YearStack, year_class: pd.Series | None = None,
                     include_mss_veg: bool = False) -> dict[str, np.ndarray]:
    wet = wet_any(st, include_mss_veg)
    obs = st.obs_spr | st.obs_late
    out = {
        "n_obs_years": obs.sum(0),
        "n_wet_years": wet.sum(0),
        "f_wet": _safe_div(wet.sum(0), obs.sum(0)),
        "f_spring": _safe_div(st.w_spr.sum(0), st.obs_spr.sum(0)),
        "f_late": _safe_div(st.w_late.sum(0), st.obs_late.sum(0)),
        "f_wetveg": _safe_div(st.wv_late.sum(0), st.obs_late.sum(0)),
    }
    if year_class is not None:
        cls = np.array([year_class.get(y, "no_data") for y in st.years])
        for label, members in (("wet_yrs", {"wet", "very_wet"}), ("dry_yrs", {"dry", "very_dry"}),
                               ("normal_yrs", {"normal"})):
            sel = np.isin(cls, list(members))
            if sel.any():
                out[f"f_late_{label}"] = _safe_div(st.w_late[sel].sum(0), st.obs_late[sel].sum(0))
                out[f"f_spring_{label}"] = _safe_div(st.w_spr[sel].sum(0), st.obs_spr[sel].sum(0))
                out[f"f_wet_{label}"] = _safe_div(wet[sel].sum(0), obs[sel].sum(0))
            else:
                for k in ("f_late", "f_spring", "f_wet"):
                    out[f"{k}_{label}"] = np.full(wet.shape[1:], np.nan)
    return out


def greatest_extent_mask(st: YearStack, thr: Thresholds, include_mss_veg: bool = False,
                         years_subset: list[int] | None = None) -> np.ndarray:
    wet = wet_any(st, include_mss_veg)
    if years_subset is not None:
        sel = np.isin(st.years, years_subset)
        wet = wet[sel]
    count = wet.sum(0)
    mask = count >= thr.min_years_for_extent
    return clean_mask(mask, st.pixel_area_m2, thr)


def clean_mask(mask: np.ndarray, pixel_area_m2: float, thr: Thresholds) -> np.ndarray:
    """Remove slivers, fill small holes, one open/close pass to smooth edges."""
    if not mask.any():
        return mask
    min_px = max(1, int(round(thr.min_wetland_area_m2 / pixel_area_m2)))
    hole_px = max(1, int(round(thr.fill_hole_area_m2 / pixel_area_m2)))
    struct = ndimage.generate_binary_structure(2, 2)
    # fill holes
    inv_lab, n = ndimage.label(~mask, structure=struct)
    if n:
        sizes = ndimage.sum(np.ones_like(mask, dtype=np.int32), inv_lab, index=np.arange(1, n + 1))
        small = np.isin(inv_lab, np.where(sizes < hole_px)[0] + 1)
        mask = mask | small
    # smooth (open then close) with a single pixel structuring element
    mask = ndimage.binary_opening(mask, structure=struct)
    mask = ndimage.binary_closing(mask, structure=struct)
    # remove slivers
    lab, n = ndimage.label(mask, structure=struct)
    if n:
        sizes = ndimage.sum(np.ones_like(mask, dtype=np.int32), lab, index=np.arange(1, n + 1))
        keep = np.where(sizes >= min_px)[0] + 1
        mask = np.isin(lab, keep)
    return mask


def yearly_extent_table(st: YearStack, extent_mask: np.ndarray, precip_years: pd.DataFrame | None,
                        include_mss_veg: bool = False) -> pd.DataFrame:
    """Area of water / wet vegetation inside the greatest extent, per year."""
    wet = wet_any(st, include_mss_veg)
    px_ha = st.pixel_area_m2 / 1e4
    rows = []
    for i, y in enumerate(st.years):
        obs = (st.obs_spr[i] | st.obs_late[i]) & extent_mask
        rows.append({
            "year": y,
            "sensor": "MSS" if st.is_mss[i] else "Landsat/S2",
            "observed_ha": obs.sum() * px_ha,
            "spring_water_ha": (st.w_spr[i] & extent_mask).sum() * px_ha,
            "late_water_ha": (st.w_late[i] & extent_mask).sum() * px_ha,
            "wet_veg_ha": (st.wv_late[i] & extent_mask).sum() * px_ha,
            "wet_total_ha": (wet[i] & extent_mask).sum() * px_ha,
            "coverage_pct": 100.0 * obs.sum() / max(extent_mask.sum(), 1),
        })
    df = pd.DataFrame(rows).set_index("year")
    if precip_years is not None:
        df = df.join(precip_years[["precip_mm", "pct_of_normal", "percentile", "wet_dry", "wet_dry_24mo", "spi12"]],
                     how="left")
    return df.round(2)


# --------------------------------------------------------------------------- #
# Vectorise and number wetlands
# --------------------------------------------------------------------------- #
def vectorise(mask: np.ndarray, transform, crs: str, smooth_m: float = 5.0) -> gpd.GeoDataFrame:
    shapes = features.shapes(mask.astype(np.uint8), mask=mask, transform=transform, connectivity=8)
    geoms = [shape(g) for g, v in shapes if v == 1]
    gdf = gpd.GeoDataFrame(geometry=geoms, crs=crs)
    if gdf.empty:
        return gdf
    if smooth_m > 0:  # buffer out/in removes the pixel staircase
        gdf["geometry"] = gdf.buffer(smooth_m, join_style=1).buffer(-smooth_m, join_style=1)
    gdf = gdf[~gdf.is_empty].explode(index_parts=False).reset_index(drop=True)
    gdf["area_ha"] = gdf.area / 1e4
    return gdf


def number_wetlands(wetlands: gpd.GeoDataFrame, project_area, rings: dict, site_prefix: str,
                    study_area=None) -> gpd.GeoDataFrame:
    """Assign stable IDs and locate each wetland relative to the project area.

    IDs are assigned inside the project area first (north-west to south-east
    so numbering reads naturally on a map), then in the rings outward, then
    the remaining study-area wetlands.

        wetland_id      e.g. "BRI-W07"
        location        project_area | within_400m | within_800m | study_area
        dist_to_project_m  0 inside, otherwise distance to the nearest edge
        pct_in_project  share of the polygon inside the project area
    """
    if wetlands.empty:
        return wetlands
    w = wetlands.copy()
    pa = gpd.GeoSeries([project_area], crs="EPSG:4326").to_crs(w.crs).iloc[0]
    w["dist_to_project_m"] = w.geometry.distance(pa).round(1)
    w["pct_in_project"] = (w.geometry.intersection(pa).area / w.geometry.area * 100).round(1)

    # rings sorted inner -> outer by area
    ring_items = sorted(rings.items(), key=lambda kv: gpd.GeoSeries([kv[1]], crs="EPSG:4326").to_crs(w.crs).area.iloc[0])
    ring_geoms = [(n, gpd.GeoSeries([g], crs="EPSG:4326").to_crs(w.crs).iloc[0]) for n, g in ring_items]

    def _loc(geom):
        if geom.intersects(pa):
            return "project_area", 0
        for k, (n, rg) in enumerate(ring_geoms, start=1):
            if geom.intersects(rg):
                return f"within_{n.split()[0]}", k
        return "study_area", len(ring_geoms) + 1

    locs = [_loc(g) for g in w.geometry]
    w["location"] = [l for l, _ in locs]
    w["_rank"] = [r for _, r in locs]
    c = w.geometry.centroid
    w["_cx"], w["_cy"] = c.x, c.y
    w = w.sort_values(["_rank", "_cy", "_cx"], ascending=[True, False, True]).reset_index(drop=True)
    w["wetland_id"] = [f"{site_prefix}-W{i+1:02d}" for i in range(len(w))]
    cen = w.geometry.centroid.to_crs("EPSG:4326")
    w["centroid_lat"] = cen.y.round(6)
    w["centroid_lon"] = cen.x.round(6)
    return w.drop(columns=["_rank", "_cx", "_cy"])


def zonal_stats(wetlands: gpd.GeoDataFrame, layers: dict[str, np.ndarray], transform,
                stats: tuple[str, ...] = ("mean",)) -> gpd.GeoDataFrame:
    """Mean of each frequency / static layer inside every wetland polygon."""
    if wetlands.empty:
        return wetlands
    shp = next(iter(layers.values())).shape
    lab = features.rasterize(((g, i + 1) for i, g in enumerate(wetlands.geometry)), out_shape=shp,
                             transform=transform, fill=0, dtype="int32", all_touched=False)
    idx = np.arange(1, len(wetlands) + 1)
    out = wetlands.copy()
    for name, arr in layers.items():
        a = arr.astype("float64")
        valid = ~np.isnan(a)
        s = ndimage.sum(np.where(valid, a, 0), lab, index=idx)
        n = ndimage.sum(valid.astype(np.int32), lab, index=idx)
        out[name] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return out


def zonal_core_stats(wetlands: gpd.GeoDataFrame, freq: dict[str, np.ndarray], transform,
                     permanent_thr: float = 0.9, seasonal_thr: float = 0.5) -> gpd.GeoDataFrame:
    """Core statistics that the polygon *mean* hides: a pond with a wide seasonal
    fringe still has a permanent core. Adds

        f_late_p90, f_spring_p90      90th percentile of frequency inside the polygon
        frac_permanent_water          share of pixels with late-season water in >= permanent_thr of years
        frac_seasonal_water           share of pixels with spring water in >= seasonal_thr of years
    """
    if wetlands.empty:
        return wetlands
    shp = freq["f_late"].shape
    lab = features.rasterize(((g, i + 1) for i, g in enumerate(wetlands.geometry)), out_shape=shp,
                             transform=transform, fill=0, dtype="int32")
    idx = np.arange(1, len(wetlands) + 1)
    out = wetlands.copy()

    def _p90(a):
        a = a[~np.isnan(a)]
        return np.nan if a.size == 0 else float(np.percentile(a, 90))

    for k in ("f_late", "f_spring"):
        out[f"{k}_p90"] = ndimage.labeled_comprehension(freq[k], lab, idx, _p90, float, np.nan)
    n = ndimage.sum(np.ones(shp, dtype=np.int32), lab, index=idx)
    perm = (np.nan_to_num(freq["f_late"], nan=0) >= permanent_thr).astype(np.int32)
    seas = (np.nan_to_num(freq["f_spring"], nan=0) >= seasonal_thr).astype(np.int32)
    out["frac_permanent_water"] = ndimage.sum(perm, lab, index=idx) / np.maximum(n, 1)
    out["frac_seasonal_water"] = ndimage.sum(seas, lab, index=idx) / np.maximum(n, 1)
    return out


def class_fraction(wetlands: gpd.GeoDataFrame, class_raster: np.ndarray, transform,
                   classes: dict[str, list[int]]) -> gpd.GeoDataFrame:
    """Fraction of each polygon covered by given raster class codes (e.g. WorldCover)."""
    if wetlands.empty:
        return wetlands
    lab = features.rasterize(((g, i + 1) for i, g in enumerate(wetlands.geometry)), out_shape=class_raster.shape,
                             transform=transform, fill=0, dtype="int32")
    idx = np.arange(1, len(wetlands) + 1)
    n = ndimage.sum(np.ones_like(class_raster, dtype=np.int32), lab, index=idx)
    out = wetlands.copy()
    for name, codes in classes.items():
        m = np.isin(class_raster, codes).astype(np.int32)
        out[f"frac_{name}"] = ndimage.sum(m, lab, index=idx) / np.maximum(n, 1)
    return out
