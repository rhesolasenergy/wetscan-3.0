"""Google Dynamic World hydroperiod metrics for WetScan.

This module is intentionally separate from the existing WetScan
imagery/classification pipeline while the hydroperiod method is
being developed and validated.
"""

from __future__ import annotations


DYNAMIC_WORLD_COLLECTION = "GOOGLE/DYNAMICWORLD/V1"


def dynamic_world_collection(region, start: str, end: str):
    """Return Dynamic World observations for a region and date range."""
    import ee

    return (
        ee.ImageCollection(DYNAMIC_WORLD_COLLECTION)
        .filterBounds(region)
        .filterDate(start, end)
    )


def _wet_probability(img):
    """Combine open-water and flooded-vegetation probabilities."""
    water = img.select("water")
    flooded = img.select("flooded_vegetation")

    return (
        water.max(flooded)
        .rename("wet_probability")
        .copyProperties(img, ["system:time_start"])
    )


def hydroperiod_metrics(
    region,
    start: str,
    end: str,
    probability_threshold: float = 0.5,
):
    """Create Dynamic World hydroperiod metrics for one period."""
    import ee

    collection = dynamic_world_collection(region, start, end)

    water = collection.select("water")
    flooded = collection.select("flooded_vegetation")

    wet_probability = collection.map(_wet_probability)

    wet_freq_10 = wet_probability.map(
        lambda img: img.gte(0.10).rename("wet")
    ).mean().rename("dw_wet_freq_10")

    wet_freq_20 = wet_probability.map(
        lambda img: img.gte(0.20).rename("wet")
    ).mean().rename("dw_wet_freq_20")

    wet_freq_30 = wet_probability.map(
        lambda img: img.gte(0.30).rename("wet")
    ).mean().rename("dw_wet_freq_30")

    water_label_frequency = collection.map(
        lambda img: img.select("label").eq(0).rename("water_label")
    ).mean().rename("dw_water_label_freq")

    flooded_label_frequency = collection.map(
        lambda img: img.select("label").eq(3).rename("flooded_label")
    ).mean().rename("dw_floodedveg_label_freq")

    return (
        ee.Image.cat(
            [
                water.mean().rename("dw_water_prob_mean"),
                flooded.mean().rename("dw_floodedveg_prob_mean"),
                wet_probability.mean().rename("dw_wet_prob_mean"),
                wet_freq_10,
                wet_freq_20,
                wet_freq_30,
                water_label_frequency,
                flooded_label_frequency,
                wet_probability.count().rename("dw_n_obs"),
            ]
        )
        .toFloat()
        .clip(region)
    )
def annual_hydroperiod_metrics(
    region,
    year: int,
    probability_threshold: float = 0.5,
):
    """Dynamic World hydroperiod metrics for the Alberta growing season.

    Uses April 20 through September 20 to match WetScan's existing
    spring/late-summer analysis period broadly. We can refine the
    seasonal definition later during validation.
    """
    import ee

    start = ee.Date.fromYMD(year, 4, 20)
    end = ee.Date.fromYMD(year, 9, 21)

    return hydroperiod_metrics(
        region,
        start.format("YYYY-MM-dd"),
        end.format("YYYY-MM-dd"),
        probability_threshold,
    )

def download_annual_hydroperiod_geotiff(
    region,
    year: int,
    out_path,
    crs: str,
    scale: int = 20,
    probability_threshold: float = 0.5,
):
    """Download annual Dynamic World hydroperiod metrics as one GeoTIFF."""
    import io
    import zipfile
    from pathlib import Path

    import requests

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    img = annual_hydroperiod_metrics(
        region,
        year,
        probability_threshold=probability_threshold,
    )

    bands = [
        "dw_water_prob_mean",
        "dw_floodedveg_prob_mean",
        "dw_wet_prob_mean",
        "dw_wet_freq_10",
        "dw_wet_freq_20",
        "dw_wet_freq_30",
        "dw_water_label_freq",
        "dw_floodedveg_label_freq",
        "dw_n_obs",
    ]

    url = img.select(bands).getDownloadURL(
        {
            "region": region,
            "scale": scale,
            "crs": crs,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )

    response = requests.get(url, timeout=600)

    if not response.ok:
        raise RuntimeError(
            f"Dynamic World download failed "
            f"({response.status_code}): {response.text[:3000]}"
        )

    if response.headers.get("content-type", "").startswith("application/zip"):
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            tif_names = [n for n in z.namelist() if n.lower().endswith(".tif")]

            if not tif_names:
                raise RuntimeError("Earth Engine ZIP contained no GeoTIFF.")

            out_path.write_bytes(z.read(tif_names[0]))
    else:
        out_path.write_bytes(response.content)

    return out_path

def build_evidence_tables(
    polygons_path,
    site: str,
    out_dir,
    start_year: int,
    end_year: int,
    gee_project: str,
    scale_m: int = 10,
    spring_start: str = "04-20",
    spring_end: str = "06-20",
    late_start: str = "07-15",
    late_end: str = "09-20",
    layer: str | None = None,
):
    """Build Dynamic World wetland-by-year and wetland-summary tables.

    Works with polygon GPKG, SHP or GeoJSON inputs.

    Outputs
    -------
    <site>_DW_evidence_by_year.csv
        One row per wetland per year.

    <site>_DW_summary_by_wetland.csv
        Multi-year Dynamic World hydroperiod summary per wetland.

    <site>_DW_run_summary.json
        Run configuration / provenance.
    """
    import json
    from pathlib import Path

    import ee
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import mapping

    polygons_path = Path(polygons_path)
    out_dir = Path(out_dir) / site
    out_dir.mkdir(parents=True, exist_ok=True)

    ee.Initialize(project=gee_project)

    # --------------------------------------------------------------
    # Read polygon layer
    # --------------------------------------------------------------
    if polygons_path.suffix.lower() == ".gpkg":
        import pyogrio

        available = pyogrio.list_layers(polygons_path)

        if layer is None:
            polygon_layers = [
                str(name)
                for name, geom_type in available
                if "polygon" in str(geom_type).lower()
            ]

            if not polygon_layers:
                raise RuntimeError(
                    f"No polygon layer found in {polygons_path}"
                )

            # Prefer a layer containing "wetland"
            wetland_layers = [
                name for name in polygon_layers
                if "wetland" in name.lower()
            ]

            layer = (
                wetland_layers[0]
                if wetland_layers
                else polygon_layers[0]
            )

        gdf = gpd.read_file(polygons_path, layer=layer)

    else:
        gdf = gpd.read_file(polygons_path)

    if gdf.empty:
        raise RuntimeError("Polygon input contains no features.")

    if gdf.crs is None:
        raise RuntimeError("Polygon input has no CRS.")

    # --------------------------------------------------------------
    # Determine wetland identifier
    # --------------------------------------------------------------
    candidates = [
        "wetland_id",
        "wetland_no",
        "wetland",
        "id",
        "name",
    ]

    lower_columns = {
        c.lower(): c for c in gdf.columns
    }

    id_col = next(
        (
            lower_columns[c]
            for c in candidates
            if c in lower_columns
        ),
        None,
    )

    gdf_wgs84 = gdf.to_crs("EPSG:4326")

    ee_features = []

    for i, row in gdf_wgs84.iterrows():
        if id_col is not None:
            wetland_id = str(row[id_col])
        else:
            wetland_id = f"{site}_W{i + 1:03d}"

        ee_features.append(
            ee.Feature(
                ee.Geometry(mapping(row.geometry)),
                {
                    "wetland_id": wetland_id,
                    "site": site,
                },
            )
        )

    wetlands = ee.FeatureCollection(ee_features)

    # --------------------------------------------------------------
    # Helper: polygon means
    # --------------------------------------------------------------
    def reduce_image(image, prefix):
        reduced = image.reduceRegions(
            collection=wetlands,
            reducer=ee.Reducer.mean(),
            scale=scale_m,
        )

        features = reduced.getInfo()["features"]

        rows = []

        for feature in features:
            props = feature["properties"]

            row = {
                "site": props.get("site"),
                "wetland_id": props.get("wetland_id"),
            }

            for key, value in props.items():
                if key in ("site", "wetland_id"):
                    continue

                row[f"{prefix}{key}"] = value

            rows.append(row)

        return rows

    # --------------------------------------------------------------
    # Generate year-by-year evidence
    # --------------------------------------------------------------
    all_rows = []

    for year in range(start_year, end_year + 1):
        print(f"Dynamic World {site}: {year}")

        annual = hydroperiod_metrics(
            wetlands.geometry(),
            f"{year}-{spring_start}",
            f"{year}-{late_end}",
        )

        spring = hydroperiod_metrics(
            wetlands.geometry(),
            f"{year}-{spring_start}",
            f"{year}-{spring_end}",
        )

        late = hydroperiod_metrics(
            wetlands.geometry(),
            f"{year}-{late_start}",
            f"{year}-{late_end}",
        )

        annual_rows = reduce_image(
            annual,
            "annual_",
        )

        spring_rows = {
            r["wetland_id"]: r
            for r in reduce_image(spring, "spring_")
        }

        late_rows = {
            r["wetland_id"]: r
            for r in reduce_image(late, "late_")
        }

        for row in annual_rows:
            wid = row["wetland_id"]

            for extra in (
                spring_rows.get(wid, {}),
                late_rows.get(wid, {}),
            ):
                row.update(
                    {
                        k: v
                        for k, v in extra.items()
                        if k not in ("site", "wetland_id")
                    }
                )

            row["year"] = year

            water = row.get(
                "annual_dw_water_prob_mean"
            )
            flooded = row.get(
                "annual_dw_floodedveg_prob_mean"
            )
            wet = row.get(
                "annual_dw_wet_prob_mean"
            )

            vals = [
                x
                for x in (water, flooded, wet)
                if x is not None
                and not pd.isna(x)
            ]

            row["dw_evidence_avg"] = (
                float(np.mean(vals))
                if vals
                else np.nan
            )

            spring_wet = row.get(
                "spring_dw_wet_prob_mean"
            )

            late_wet = row.get(
                "late_dw_wet_prob_mean"
            )

            if (
                spring_wet is not None
                and late_wet is not None
                and spring_wet > 0
            ):
                row["dw_seasonal_persistence"] = (
                    late_wet / spring_wet
                )
            else:
                row["dw_seasonal_persistence"] = np.nan

            all_rows.append(row)

    yearly = pd.DataFrame(all_rows)

    yearly = yearly.sort_values(
        ["wetland_id", "year"]
    )

    yearly_path = (
        out_dir
        / f"{site}_DW_evidence_by_year.csv"
    )

    yearly.to_csv(
        yearly_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Multi-year summary per wetland
    # --------------------------------------------------------------
    summary_rows = []

    for wetland_id, grp in yearly.groupby("wetland_id"):
        wet = grp["annual_dw_wet_prob_mean"]

        spring = grp["spring_dw_wet_prob_mean"]
        late = grp["late_dw_wet_prob_mean"]

        summary_rows.append(
            {
                "site": site,
                "wetland_id": wetland_id,

                "dw_n_years": int(len(grp)),

                "dw_mean_water_prob":
                    grp[
                        "annual_dw_water_prob_mean"
                    ].mean(),

                "dw_mean_floodedveg_prob":
                    grp[
                        "annual_dw_floodedveg_prob_mean"
                    ].mean(),

                "dw_mean_wet_prob":
                    wet.mean(),

                "dw_mean_evidence":
                    grp["dw_evidence_avg"].mean(),

                "dw_wet_prob_sd":
                    wet.std(),

                "dw_wet_prob_min":
                    wet.min(),

                "dw_wet_prob_max":
                    wet.max(),

                "dw_mean_spring_wet":
                    spring.mean(),

                "dw_mean_late_wet":
                    late.mean(),

                "dw_mean_seasonal_persistence":
                    grp[
                        "dw_seasonal_persistence"
                    ].mean(),

                "dw_late_minus_spring":
                    (late - spring).mean(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        out_dir
        / f"{site}_DW_summary_by_wetland.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------------
    # Run metadata
    # --------------------------------------------------------------
    run_summary = {
        "site": site,
        "polygon_file": str(polygons_path),
        "polygon_layer": layer,
        "wetlands": int(len(gdf)),
        "start_year": start_year,
        "end_year": end_year,
        "spring_window": [
            spring_start,
            spring_end,
        ],
        "late_window": [
            late_start,
            late_end,
        ],
        "scale_m": scale_m,
        "yearly_table": str(yearly_path),
        "summary_table": str(summary_path),
    }

    summary_json = (
        out_dir
        / f"{site}_DW_run_summary.json"
    )

    summary_json.write_text(
        json.dumps(
            run_summary,
            indent=2,
        )
    )

    return {
        "site": site,
        "wetlands": len(gdf),
        "years": end_year - start_year + 1,
        "yearly_table": str(yearly_path),
        "summary_table": str(summary_path),
        "run_summary": str(summary_json),
    }