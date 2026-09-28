"""Estimated Alberta Wetland Classification System (AWCS) class and
Stewart & Kantrud (1971) permanence class for every wetland polygon.

This is a *desktop screening* estimate from multi-decade optical imagery.
It cannot see peat depth, water chemistry or water depth, so:

  * Bog vs fen cannot be separated spectrally - peatlands are reported as
    "Fen (bog possible)" and should be confirmed in the field.
  * Shallow open water is assigned where open water persists into late
    summer in most years; depth (< 2 m) is assumed, not measured.
  * Swamp vs treed peatland uses the seasonal-flooding signal: swamps flood
    in spring and dry down, peatlands stay saturated with little open water.

Inputs per wetland (columns produced by extent.zonal_stats / class_fraction):

  f_spring, f_late, f_wetveg, f_wet          inundation / wetness frequencies
  f_late_wet_yrs, f_late_dry_yrs             late-season water freq. in wet/dry years
  frac_tree, frac_shrub, frac_herb_wet, frac_water, frac_crop   ESA WorldCover shares
  treecover_pct, jrc_occurrence, slope_deg   priors
  natural_region                             grassland | parkland | boreal | foothills

Outputs:

  awcs_class        Shallow Open Water | Marsh | Swamp | Fen (bog possible) | Bog/Fen (peatland)
  awcs_form         open water / graminoid / shrubby / wooded (AWCS "form" level)
  sk_class          I ephemeral, II temporary, III seasonal, IV semi-permanent, V permanent
  permanence        text label for sk_class
  confidence        high / medium / low
  rationale         one-line explanation of which rules fired
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Thresholds

WORLDCOVER = {
    "tree": [10],
    "shrub": [20],
    "grass": [30],
    "crop": [40],
    "built": [50],
    "bare": [60],
    "water": [80],
    "herb_wet": [90],
    "moss": [100],
}

PEATLAND_REGIONS = {"boreal", "foothills", "canadian_shield", "rocky_mountain"}

SK_LABEL = {1: "ephemeral", 2: "temporary", 3: "seasonal", 4: "semi-permanent", 5: "permanent"}


def natural_region_from_lat(lat: float) -> str:
    """Coarse fallback when the user does not supply the Alberta Natural Region.
    Boundaries are approximate latitudes through central Alberta and should be
    overridden with the official Natural Regions & Subregions layer if available."""
    if lat < 52.3:
        return "grassland"
    if lat < 54.3:
        return "parkland"
    return "boreal"


def _sk_class(f_spring, f_late, thr: Thresholds) -> int:
    f_spring = 0.0 if np.isnan(f_spring) else f_spring
    f_late = 0.0 if np.isnan(f_late) else f_late
    if f_late >= thr.permanent_late_freq:
        return 5
    if f_late >= thr.semiperm_late_freq:
        return 4
    if f_spring >= thr.seasonal_spring_freq:
        return 3
    if f_spring >= 0.15:
        return 2
    return 1


def classify_row(r: pd.Series, thr: Thresholds) -> dict:
    g = lambda k, d=np.nan: float(r[k]) if k in r and pd.notna(r[k]) else d  # noqa: E731
    region = str(r.get("natural_region", "parkland")).lower()
    peat_region = region in PEATLAND_REGIONS

    f_spring, f_late, f_wetveg, f_wet = g("f_spring", 0), g("f_late", 0), g("f_wetveg", 0), g("f_wet", 0)
    f_late_wet, f_late_dry = g("f_late_wet_yrs"), g("f_late_dry_yrs")
    woody = g("frac_tree", 0) + g("frac_shrub", 0)
    tree = g("frac_tree", 0)
    herb_wet = g("frac_herb_wet", 0)
    crop = g("frac_crop", 0)
    jrc = g("jrc_occurrence", 0)
    open_frac = g("frac_water", 0)

    # permanence is judged on the wettest core of the basin (90th percentile),
    # not the polygon mean, because the greatest extent includes the fringe
    f_spring_core = g("f_spring_p90", f_spring)
    f_late_core = g("f_late_p90", f_late)
    frac_perm = g("frac_permanent_water", 0)
    sk = _sk_class(f_spring_core, f_late_core, thr)
    notes = [f"S&K {sk} (core: spring water {f_spring_core:.0%} of yrs, late water {f_late_core:.0%}; "
             f"polygon mean spring {f_spring:.0%}, late {f_late:.0%})"]

    # variability of late-season water between wet and dry years:
    # peatlands and permanent basins barely change; marshes/swamps swing.
    swing = np.nan
    if not np.isnan(f_late_wet) and not np.isnan(f_late_dry):
        swing = f_late_wet - f_late_dry
        notes.append(f"late water wet-yrs {f_late_wet:.0%} vs dry-yrs {f_late_dry:.0%}")

    confidence = "medium"
    open_share = max(frac_perm, open_frac)
    if sk == 5 and open_share >= thr.open_water_frac_sow:
        cls, form = "Shallow Open Water", "open water"
        notes.append(f"permanent open water over {open_share:.0%} of the basin"
                     + ("; marsh fringe likely" if open_share < 0.75 else ""))
        confidence = "high" if jrc >= 50 else "medium"
    elif sk == 5 and open_share > 0.1 and woody < thr.woody_frac_swamp:
        cls, form = "Marsh", "graminoid"
        notes.append(f"permanent open-water zone ({open_share:.0%}) inside a larger emergent marsh")
    elif woody >= thr.woody_frac_swamp:
        if peat_region and f_late < 0.2 and f_spring_core < 0.3 and (np.isnan(swing) or abs(swing) < 0.25) and f_wetveg >= 0.5:
            cls, form = "Fen (bog possible)", "wooded" if tree > 0.4 else "shrubby"
            notes.append("wooded, saturated all years, no seasonal flooding -> treed peatland")
            confidence = "low"
        else:
            cls, form = "Swamp", "wooded" if tree > 0.4 else "shrubby"
            notes.append("woody cover with seasonal surface water")
    else:
        if peat_region and f_late < 0.2 and f_spring_core < 0.3 and f_wetveg >= 0.6 and (np.isnan(swing) or abs(swing) < 0.25) and crop < 0.2:
            cls, form = "Fen (bog possible)", "graminoid"
            notes.append("herbaceous, persistently saturated, little open water, low wet/dry swing -> peatland")
            confidence = "low"
        elif sk >= 3 or herb_wet >= 0.3 or f_wetveg >= 0.3:
            cls, form = "Marsh", "graminoid"
            notes.append("emergent vegetation with seasonal to semi-permanent water")
            if crop >= 0.5:
                notes.append("mostly cropped in WorldCover: cultivated temporary wetland")
                confidence = "low"
        else:
            cls, form = "Marsh", "graminoid"
            notes.append("ephemeral/temporary depression (S&K I-II); may be a cultivated wetland")
            confidence = "low"

    # confidence adjustments
    n_obs = g("n_obs_years", 0)
    if n_obs < 8:
        confidence = "low"
        notes.append(f"only {n_obs:.0f} observed years")
    if g("slope_deg", 0) > 6:
        confidence = "low"
        notes.append("steep terrain: check for shadow/false positive")

    return {
        "awcs_class": cls,
        "awcs_form": form,
        "sk_class": sk,
        "permanence": SK_LABEL[sk],
        "confidence": confidence,
        "rationale": "; ".join(notes),
    }


def classify_wetlands(wetlands: pd.DataFrame, thr: Thresholds = Thresholds()) -> pd.DataFrame:
    if wetlands.empty:
        return wetlands
    recs = [classify_row(r, thr) for _, r in wetlands.iterrows()]
    out = wetlands.copy()
    for k in recs[0]:
        out[k] = [r[k] for r in recs]
    return out


def summarise(wetlands: pd.DataFrame) -> pd.DataFrame:
    """Count and area by AWCS class and location for the report."""
    if wetlands.empty:
        return pd.DataFrame(columns=["location", "awcs_class", "count", "area_ha"])
    s = (wetlands.groupby(["location", "awcs_class"])
         .agg(count=("wetland_id", "count"), area_ha=("area_ha", "sum"))
         .reset_index())
    s["area_ha"] = s["area_ha"].round(2)
    return s
