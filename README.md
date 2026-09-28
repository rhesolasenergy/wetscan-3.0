# wetscan

Desktop wetland screening for Alberta project sites: **greatest wetland extent**
over the full satellite record (1972-present), an **estimated Alberta Wetland
Classification System (AWCS) class** and Stewart & Kantrud permanence class for
every wetland, with every imagery year tagged **wet / normal / dry** from
historical precipitation so you can see which years drove the maximum.

Give it a KMZ of the project area (the `Portage Power - Six AOIs` file works as
is) and it produces, per site:

| Output | What it is |
|---|---|
| `<site>_wetlands.gpkg` (+ `.geojson`, `.kml`, `.csv`) | Numbered wetland polygons (`BRI-W01`, `BRI-W02`, ...) with AWCS class, permanence, area, wet-year and dry-year extent, location relative to the project area (inside / within 400 m / within 800 m / study area), inundation frequencies, land-cover priors, confidence and a one-line rationale |
| `greatest_extent.tif` | Union of every year's water + wet-vegetation mask (uint8) |
| `inundation_frequency.tif` | Fraction of observed years each pixel was wet |
| `extent_by_year.csv` | Wet area inside the greatest extent for every year, joined to precipitation and wet/dry class |
| `precip_years.csv`, `precip_monthly.csv` | Water-year (Oct-Sep) totals, % of 1991-2020 normal, percentile, SPI-12/24, wet/dry class |
| `masks/masks_<year>.tif` | Cached per-year masks (spring water, late-summer water, wet vegetation, observation counts) |
| `basemaps/google_z17.tif`, `azure_z17.tif` | Optional high-resolution mosaics for visual QA / digitising |
| `<site>_report.html` | Self-contained report: precipitation chart, wet area by year, map with numbered wetlands, classification table |

## How it works

1. **AOI** - KMZ/KML folders become sites. Placemarks named like `800m Notification`
   or `400m Consultation` are kept as reference rings; the other placemarks
   (quarter sections) are dissolved into the *project area*. The *study area*
   is the project area buffered by `--study-buffer-m` (default **2000 m**).
2. **Precipitation** - ERA5-Land monthly totals (Earth Engine, 1950-present) or
   ECCC climate stations (`--precip-source eccc`, no Earth Engine needed).
   Each water year is classified against the 1991-2020 normal:
   very dry <= P10, dry <= P20, normal, wet >= P80, very wet >= P90. The
   24-month antecedent total is used where available because wetland water
   levels lag rainfall by a season or more.
3. **Imagery** - every cloud-screened Landsat MSS/TM/ETM+/OLI and Sentinel-2
   scene is harmonised and two composites are built per year: a spring window
   (Apr 20-Jun 20, 75th-percentile water index = maximum inundation) and a
   late-summer window (Jul 15-Sep 20, median = persistence). The *index values*
   (MNDWI, NDMI, NDVI) are downloaded as small int16 GeoTIFFs
   (`masks/indices_<year>.tif`) so all thresholding happens locally, is cached,
   and can be re-tuned with `--skip-download` in seconds.

   **What counts as wet (v0.2).** Open water: MNDWI > 0 (NDWI > 0.1 for MSS).
   Wet vegetation is deliberately strict, because "green and moist in August"
   describes every good crop: a pixel qualifies only if its late-summer NDMI is
   at least 0.12 above the mean NDMI of the surrounding 500 m (a wetland basin is
   an anomaly in its landscape; a field is not), NDMI > 0.30 and NDVI > 0.30, it
   does so in at least half of the observed years (crops rotate, wetlands do
   not), and it lies within 100 m of a pixel that has held open water in any
   year or is mapped as water / herbaceous wetland by ESA WorldCover. WorldCover
   cropland and built-up pixels and slopes over 5 degrees never qualify through
   the vegetation rule (a cultivated temporary wetland is still caught when it
   ponds in spring). In boreal / foothills regions a peatland exception keeps
   persistently moist (NDMI > 0.35) tree, shrub or herbaceous-wetland pixels
   without nearby open water, because fens and bogs rarely show open water.
   v0.1 used an absolute NDMI > 0.2 rule and mapped lush cropland as wetland;
   v0.1 mask caches are not compatible - delete `masks/` or run `--overwrite`.
4. **Greatest extent** - a pixel enters the extent if it was wet in at least
   `--min-years` years (default 2, which removes single-scene noise). Slivers
   under `--min-area-ha` are dropped, holes filled, edges smoothed, and the
   result vectorised. Wetlands are numbered inside the project area first
   (NW to SE), then outward through the rings, then the rest of the study area.
5. **Classification** - per wetland, from inundation frequencies (core = 90th
   percentile, so a pond with a marsh fringe is still a pond), ESA WorldCover
   tree/shrub/crop/herbaceous-wetland shares, JRC surface-water occurrence,
   slope and the Natural Region:

   | Signal | Estimated class |
   |---|---|
   | late-summer open water in >= 90 % of years over >= 50 % of the basin | Shallow Open Water (S&K V) |
   | woody cover >= 40 % with seasonal surface water | Swamp |
   | woody or graminoid, boreal/foothills, saturated every year, almost never flooded, little wet/dry swing | Fen (bog possible) |
   | emergent vegetation with seasonal to semi-permanent water | Marsh |
   | spring water only in wet years | Marsh, S&K I-II (often cultivated) |

   Permanence: V permanent (late water >= 90 % yrs), IV semi-permanent (>= 50 %),
   III seasonal (spring water >= 50 % yrs), II temporary (>= 15 %), I ephemeral.

The classes are screening estimates. Bog vs fen, water depth and peat cannot be
seen from optical imagery; every wetland carries a confidence flag and rationale
so a wetland specialist can review before field verification under the Alberta
Wetland Identification and Delineation Directive.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

Earth Engine (free for non-commercial; commercial use needs a Google Cloud
project with Earth Engine enabled - https://earthengine.google.com/):

```bash
earthengine authenticate          # opens a browser once
```

Create a `.env` in the folder you run from (never commit it):

```
GEE_PROJECT=your-cloud-project-id
GOOGLE_MAPS_API_KEY=...        # Maps Static API, for Google satellite basemaps (optional)
AZURE_MAPS_KEY=...             # Azure Maps, the successor to Bing aerial imagery (optional)
BING_MAPS_KEY=...              # only if you still hold an active Bing Maps Enterprise key
```

Bing Maps for Enterprise was retired for new/basic keys in June 2025; Microsoft
serves the same aerial imagery through Azure Maps (`tilesetId=microsoft.imagery`),
which is what the `azure` provider uses. Google and Bing/Azure basemaps are undated
mosaics, so they are used for QA of the *current* extent, not the time series.

## Run

```bash
wetscan sites "Portage Power - Six AOIs 11SEP2026.kmz"          # list the six sites
wetscan run   "Portage Power - Six AOIs 11SEP2026.kmz" --out portage_wetlands \
              --basemap google --basemap azure                    # full run, all sites
wetscan run   ... --site Brighthold_Site134 --site Polaris         # subset
wetscan run   ... --skip-download --min-years 3                    # re-classify from cached masks
wetscan precip --lon -112.198 --lat 51.466 --source eccc           # precipitation only, no GEE
```

Options worth knowing: `--natural-region boreal|parkland|grassland|foothills`
(default is inferred from latitude; set it explicitly for Northreach/Polaris),
`--scale-m 20` analysis resolution (default 20 m; 10 m is Sentinel-2 native but four times the download), `--start-year 1984` to skip the 60 m MSS
era, `--include-mss-veg` to let 1970s wet-vegetation masks count toward extent.

Years with no cloud-free scenes or a download error are listed in
`masks/years_failed.json` and in the report header. A full run downloads ~55 small rasters per site; expect 5-15 minutes per site
depending on Earth Engine load. Re-runs with `--skip-download` take seconds.

## Tests

```bash
pip install -e .[dev]
pytest
```

The tests build a synthetic site (permanent pond with a seasonal fringe, seasonal
marsh, treed peatland, ephemeral depression, cloudy years, MSS-era years) and run
the complete offline pipeline, so the numbering, extent, precipitation
classification and AWCS rules are exercised without Earth Engine access.

## Tuning

All thresholds live in `wetscan/config.py` (`Thresholds`, `Season`, `Precip`).
The seasonal windows are set for central Alberta; shift them ~2 weeks later for
the Peace Region sites.
