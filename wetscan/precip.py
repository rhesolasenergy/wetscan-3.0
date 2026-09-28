"""Historical precipitation and wet / normal / dry year classification.

Two independent sources are supported so the tool still runs when Earth
Engine is unavailable:

* ``era5_land_monthly``  - ECMWF ERA5-Land monthly aggregates via Earth Engine
                           (0.1 deg, 1950-present). Default. Works anywhere.
* ``eccc_station``       - Environment and Climate Change Canada daily
                           observations from the nearest climate station via
                           the MSC GeoMet OGC API (no key required).

Both produce a DataFrame indexed by water year with these columns:

    precip_mm        total precipitation over the water year (Oct-Sep)
    spring_mm        Oct-Jun total (snow accumulation + spring rain; drives
                     the spring inundation maximum in prairie wetlands)
    precip_24mo_mm   two-water-year total (antecedent conditions)
    pct_of_normal    precip_mm / mean of the 1991-2020 normal period * 100
    percentile       empirical percentile of precip_mm within the normal period
    spi12            12-month Standardized Precipitation Index (gamma fit)
    spi24            24-month SPI
    wet_dry          one of very_dry / dry / normal / wet / very_wet
    n_months         months with data (flag partial years)
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from scipy import stats

from .config import Precip as PrecipCfg

CLASS_ORDER = ["very_dry", "dry", "normal", "wet", "very_wet"]


# --------------------------------------------------------------------------- #
# Source 1: ERA5-Land via Earth Engine
# --------------------------------------------------------------------------- #
def era5_land_monthly(lon: float, lat: float, start_year: int, end_year: int,
                      buffer_km: float = 5.0) -> pd.Series:
    """Monthly total precipitation (mm) averaged over a small buffer around a point."""
    import ee  # imported lazily so the module loads without GEE installed

    region = ee.Geometry.Point([lon, lat]).buffer(buffer_km * 1000)
    ic = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
          .select("total_precipitation_sum")
          .filterDate(f"{start_year - 2}-01-01", f"{end_year + 1}-01-01"))

    def _stat(img):
        v = img.reduceRegion(ee.Reducer.mean(), region, 11132).get("total_precipitation_sum")
        return ee.Feature(None, {"date": img.date().format("YYYY-MM"), "p": v})

    feats = ic.map(_stat).getInfo()["features"]
    rows = [(f["properties"]["date"], f["properties"]["p"]) for f in feats
            if f["properties"].get("p") is not None]
    s = pd.Series({pd.Period(d, "M"): p * 1000.0 for d, p in rows}).sort_index()  # m -> mm
    s.name = "precip_mm"
    return s


# --------------------------------------------------------------------------- #
# Source 2: ECCC climate stations (GeoMet OGC API)
# --------------------------------------------------------------------------- #
GEOMET = "https://api.weather.gc.ca/collections"


def eccc_nearest_stations(lon: float, lat: float, radius_km: float = 75.0, n: int = 5) -> pd.DataFrame:
    d = radius_km / 111.0
    r = requests.get(f"{GEOMET}/climate-stations/items",
                     params={"f": "json", "limit": 500,
                             "bbox": f"{lon-d},{lat-d},{lon+d},{lat+d}"},
                     timeout=60)
    r.raise_for_status()
    recs = []
    for f in r.json().get("features", []):
        p = f["properties"]
        slon, slat = f["geometry"]["coordinates"][:2]
        dist = math.hypot((slon - lon) * 111 * math.cos(math.radians(lat)), (slat - lat) * 111)
        recs.append({
            "climate_id": p.get("CLIMATE_IDENTIFIER"),
            "name": p.get("STATION_NAME"),
            "first_year": p.get("FIRST_DATE", "")[:4],
            "last_year": p.get("LAST_DATE", "")[:4],
            "has_monthly": p.get("HAS_MONTHLY_SUMMARY"),
            "dist_km": round(dist, 1),
        })
    df = pd.DataFrame(recs).sort_values("dist_km")
    return df.head(n).reset_index(drop=True)


def eccc_station_monthly(climate_id: str, start_year: int, end_year: int) -> pd.Series:
    """Monthly precipitation from ECCC daily observations (summed by month).

    Uses the *climate-daily* collection so that gaps are visible; a month with
    fewer than 25 days of observations is dropped."""
    frames = []
    for y in range(start_year - 2, end_year + 1):
        r = requests.get(f"{GEOMET}/climate-daily/items",
                         params={"f": "json", "limit": 10000,
                                 "CLIMATE_IDENTIFIER": climate_id,
                                 "datetime": f"{y}-01-01/{y}-12-31",
                                 "properties": "LOCAL_DATE,TOTAL_PRECIPITATION"},
                         timeout=120)
        if r.status_code != 200:
            continue
        rows = [(f["properties"]["LOCAL_DATE"][:10], f["properties"].get("TOTAL_PRECIPITATION"))
                for f in r.json().get("features", [])]
        if rows:
            frames.append(pd.DataFrame(rows, columns=["date", "p"]))
    if not frames:
        raise RuntimeError(f"No ECCC daily data for station {climate_id}")
    df = pd.concat(frames)
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna(subset=["p"])
    df["period"] = df["date"].dt.to_period("M")
    g = df.groupby("period")["p"]
    monthly = g.sum()[g.count() >= 25]
    monthly.name = "precip_mm"
    return monthly.sort_index()


def combine_station_records(stations: list[pd.Series]) -> pd.Series:
    """Fill gaps in the nearest station's record from the next-nearest ones."""
    if not stations:
        raise ValueError("no station records")
    out = stations[0].copy()
    for s in stations[1:]:
        out = out.combine_first(s)
    return out.sort_index()


# --------------------------------------------------------------------------- #
# Water-year aggregation and classification
# --------------------------------------------------------------------------- #
def _water_year(period: pd.Period, start_month: int) -> int:
    return period.year + 1 if period.month >= start_month else period.year


def spi(series: pd.Series, window: int) -> pd.Series:
    """Standardized Precipitation Index using a gamma fit on the rolling sum
    (McKee et al. 1993). Zero-inflation handled per the usual mixed
    distribution. Fitted on the full record for stability."""
    roll = series.rolling(window, min_periods=window).sum()
    x = roll.dropna()
    if len(x) < 10:
        return pd.Series(np.nan, index=series.index)
    q = (x <= 0).mean()
    pos = x[x > 0]
    a, loc, scale = stats.gamma.fit(pos, floc=0)
    cdf = q + (1 - q) * stats.gamma.cdf(x, a, loc=0, scale=scale)
    cdf = cdf.clip(1e-6, 1 - 1e-6)
    z = pd.Series(stats.norm.ppf(cdf), index=x.index)
    return z.reindex(series.index)


def classify_years(monthly: pd.Series, cfg: PrecipCfg = PrecipCfg()) -> pd.DataFrame:
    monthly = monthly.dropna().sort_index()
    df = monthly.to_frame("p")
    df["wy"] = [_water_year(p, cfg.water_year_start_month) for p in df.index]
    df["spring_window"] = [(p.month >= cfg.water_year_start_month) or (p.month <= 6) for p in df.index]

    spi12 = spi(monthly, 12)
    spi24 = spi(monthly, 24)

    g = df.groupby("wy")
    out = pd.DataFrame({
        "precip_mm": g["p"].sum(),
        "spring_mm": df[df.spring_window].groupby("wy")["p"].sum(),
        "n_months": g["p"].count(),
    })
    out["precip_24mo_mm"] = out["precip_mm"] + out["precip_mm"].shift(1)

    # SPI at the end of the water year (September of wy)
    def _spi_at(s, wy):
        key = pd.Period(f"{wy}-{cfg.water_year_start_month - 1:02d}", "M")
        return s.get(key, np.nan)
    out["spi12"] = [_spi_at(spi12, wy) for wy in out.index]
    out["spi24"] = [_spi_at(spi24, wy) for wy in out.index]

    complete = out[out.n_months >= 12]
    normal = complete.loc[(complete.index >= cfg.normal_start) & (complete.index <= cfg.normal_end), "precip_mm"]
    if len(normal) < 10:  # fall back to whatever complete years exist
        normal = complete["precip_mm"]
    out["pct_of_normal"] = out["precip_mm"] / normal.mean() * 100.0
    out["percentile"] = [stats.percentileofscore(normal, v, kind="mean") if n >= 12 else np.nan
                         for v, n in zip(out["precip_mm"], out["n_months"])]

    def _cls(pct):
        if np.isnan(pct):
            return "no_data"
        if pct <= cfg.very_dry_pct:
            return "very_dry"
        if pct <= cfg.dry_pct:
            return "dry"
        if pct >= cfg.very_wet_pct:
            return "very_wet"
        if pct >= cfg.wet_pct:
            return "wet"
        return "normal"

    out["wet_dry"] = out["percentile"].map(_cls)
    # antecedent classification uses the 24-month total (wetland levels lag rainfall)
    normal24 = (complete["precip_mm"] + complete["precip_mm"].shift(1)).loc[normal.index].dropna()
    out["percentile_24mo"] = [stats.percentileofscore(normal24, v, kind="mean") if not np.isnan(v) else np.nan
                              for v in out["precip_24mo_mm"]]
    out["wet_dry_24mo"] = out["percentile_24mo"].map(_cls)
    out.index.name = "water_year"
    return out.round(2)


@dataclass
class PrecipResult:
    source: str
    monthly: pd.Series
    years: pd.DataFrame
    meta: dict


def get_precip(lon: float, lat: float, start_year: int, end_year: int,
               source: str = "era5", cfg: PrecipCfg = PrecipCfg(),
               station_id: str | None = None) -> PrecipResult:
    if source == "era5":
        monthly = era5_land_monthly(lon, lat, start_year, end_year)
        meta = {"dataset": "ECMWF/ERA5_LAND/MONTHLY_AGGR", "lon": lon, "lat": lat}
    elif source == "eccc":
        if station_id:
            st = pd.DataFrame([{"climate_id": station_id, "name": station_id, "dist_km": np.nan}])
        else:
            st = eccc_nearest_stations(lon, lat)
        records = []
        for _, row in st.iterrows():
            try:
                records.append(eccc_station_monthly(row.climate_id, start_year, end_year))
            except Exception:  # noqa: BLE001 - keep trying other stations
                continue
        monthly = combine_station_records(records)
        meta = {"dataset": "ECCC climate-daily (GeoMet)", "stations": st.to_dict("records")}
    else:
        raise ValueError("source must be 'era5' or 'eccc'")
    years = classify_years(monthly, cfg)
    years = years[(years.index >= start_year) & (years.index <= end_year)]
    return PrecipResult(source=source, monthly=monthly, years=years, meta=meta)
