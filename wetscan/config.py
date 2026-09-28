"""Runtime configuration: API keys, thresholds, seasonal windows.

All keys are read from environment variables (or a .env file in the working
directory) so that nothing secret is ever committed alongside the code.

    GEE_PROJECT            Google Cloud project registered for Earth Engine
    GOOGLE_MAPS_API_KEY    Maps Static API key (satellite basemap)
    AZURE_MAPS_KEY         Azure Maps subscription key (Bing imagery successor)
    BING_MAPS_KEY          Legacy Bing Maps key (only works for enterprise
                           licences that were extended past June 2025)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass
class Keys:
    gee_project: str | None = field(default_factory=lambda: os.getenv("GEE_PROJECT"))
    google_maps: str | None = field(default_factory=lambda: os.getenv("GOOGLE_MAPS_API_KEY"))
    azure_maps: str | None = field(default_factory=lambda: os.getenv("AZURE_MAPS_KEY"))
    bing_maps: str | None = field(default_factory=lambda: os.getenv("BING_MAPS_KEY"))


@dataclass
class Season:
    """Day-of-year windows used to build per-year composites (Alberta defaults).

    spring   : snowmelt / spring runoff peak - captures maximum inundation
    late     : late growing season - captures persistence of surface water
               and is the best window for vegetation indices
    """

    spring_start: str = "04-20"
    spring_end: str = "06-20"
    late_start: str = "07-15"
    late_end: str = "09-20"


@dataclass
class Thresholds:
    """Spectral and geometric thresholds. Tunable per project; defaults are
    conservative values from the prairie-pothole / boreal literature."""

    # open water
    mndwi_water: float = 0.0       # Landsat TM/ETM+/OLI & Sentinel-2 (SWIR available)
    ndwi_water_mss: float = 0.10   # Landsat MSS (no SWIR -> McFeeters NDWI)
    # wet / saturated vegetation (emergent marsh, fen, swamp understorey).
    # A pixel must be wetter than its surroundings, not merely green:
    ndmi_anomaly: float = 0.12     # late NDMI minus mean NDMI within wetveg_window_m
    wetveg_window_m: float = 500.0
    ndmi_wet: float = 0.30         # absolute floor on late-summer NDMI
    ndmi_peat: float = 0.35        # boreal/foothills peatland exception: persistent NDMI under tree/shrub/wetland cover
    ndvi_veg_min: float = 0.30
    wetveg_min_persistence: float = 0.5   # anomalous in >= this share of observed years
    water_proximity_m: float = 100.0      # must be this close to open water (any year)
    max_slope_deg: float = 5.0
    # cloud
    max_cloud_pct: float = 60.0
    # geometry (map units = metres)
    min_years_for_extent: int = 2      # pixel must be wet in >= N years to count
    min_wetland_area_m2: float = 500.0 # drop slivers smaller than this (0.05 ha)
    fill_hole_area_m2: float = 2000.0
    smooth_buffer_m: float = 5.0
    # classification
    permanent_late_freq: float = 0.90   # S&K class V
    semiperm_late_freq: float = 0.50    # S&K class IV
    seasonal_spring_freq: float = 0.50  # S&K class III
    woody_frac_swamp: float = 0.40
    open_water_frac_sow: float = 0.50
    peatland_cv_max: float = 0.35       # low interannual variability of wetness


@dataclass
class Precip:
    normal_start: int = 1991
    normal_end: int = 2020
    water_year_start_month: int = 10  # Oct-Sep hydrological year
    dry_pct: float = 20.0
    wet_pct: float = 80.0
    very_dry_pct: float = 10.0
    very_wet_pct: float = 90.0


@dataclass
class Config:
    keys: Keys = field(default_factory=Keys)
    season: Season = field(default_factory=Season)
    thresholds: Thresholds = field(default_factory=Thresholds)
    precip: Precip = field(default_factory=Precip)
    scale_m: int = 20          # analysis grid; 10 m is possible but 4x the download
    start_year: int = 1972     # Landsat-1 MSS
    end_year: int | None = None
    working_crs: str | None = None  # default: UTM zone of AOI centroid
