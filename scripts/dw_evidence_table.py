from __future__ import annotations

import json
from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from shapely.geometry import mapping

from wetscan.dynamic_world import hydroperiod_metrics


GEE_PROJECT = "solas-wetlands-1234567"

SITE = "Brighthold_Site134"

GPKG = Path(
    "portage_wetlands"
) / SITE / f"{SITE}_wetlands.gpkg"

OUT_DIR = Path("dev_dynamic_world") / SITE
OUT_CSV = OUT_DIR / f"{SITE}_DW_evidence_by_year.csv"

START_YEAR = 2016
END_YEAR = 2026

SCALE_M = 10


def find_wetland_layer(gpkg: Path) -> str:
    layers = pyogrio.list_layers(gpkg)

    for name, geom_type in layers:
        if "wetland" in str(name).lower() and "polygon" in str(geom_type).lower():
            return str(name)

    for name, geom_type in layers:
        if "polygon" in str(geom_type).lower():
            return str(name)

    raise RuntimeError("Could not find a polygon layer in the GeoPackage.")


def choose_id_column(gdf: gpd.GeoDataFrame) -> str | None:
    candidates = [
        "wetland_id",
        "wetland_no",
        "wetland",
        "id",
        "name",
    ]

    lower = {c.lower(): c for c in gdf.columns}

    for candidate in candidates:
        if candidate in lower:
            return lower[candidate]

    return None


def polygons_to_ee(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    gdf = gdf.to_crs("EPSG:4326")

    id_col = choose_id_column(gdf)

    features = []

    for i, row in gdf.iterrows():
        geom_json = mapping(row.geometry)

        if id_col:
            wetland_id = str(row[id_col])
        else:
            wetland_id = f"{SITE}_W{i + 1:03d}"

        feature = ee.Feature(
            ee.Geometry(geom_json),
            {
                "wetland_id": wetland_id,
                "site": SITE,
            },
        )

        features.append(feature)

    return ee.FeatureCollection(features)


def reduce_metrics(
    image: ee.Image,
    wetlands: ee.FeatureCollection,
    prefix: str,
) -> list[dict]:
    reduced = image.reduceRegions(
        collection=wetlands,
        reducer=ee.Reducer.mean(),
        scale=SCALE_M,
    )

    data = reduced.getInfo()["features"]

    rows = []

    for feature in data:
        props = feature["properties"]

        row = {
            "wetland_id": props.get("wetland_id"),
            "site": props.get("site"),
        }

        for key, value in props.items():
            if key in ("wetland_id", "site"):
                continue

            row[f"{prefix}{key}"] = value

        rows.append(row)

    return rows


def merge_metric_rows(
    annual_rows: list[dict],
    spring_rows: list[dict],
    late_rows: list[dict],
    year: int,
) -> list[dict]:
    spring_lookup = {
        r["wetland_id"]: r
        for r in spring_rows
    }

    late_lookup = {
        r["wetland_id"]: r
        for r in late_rows
    }

    output = []

    for annual in annual_rows:
        wid = annual["wetland_id"]

        row = dict(annual)

        spring = spring_lookup.get(wid, {})
        late = late_lookup.get(wid, {})

        row.update({
            k: v
            for k, v in spring.items()
            if k not in ("wetland_id", "site")
        })

        row.update({
            k: v
            for k, v in late.items()
            if k not in ("wetland_id", "site")
        })

        row["year"] = year

        water = row.get("annual_dw_water_prob_mean")
        flooded = row.get("annual_dw_floodedveg_prob_mean")
        wet = row.get("annual_dw_wet_prob_mean")

        vals = [
            x for x in (water, flooded, wet)
            if x is not None and not pd.isna(x)
        ]

        row["dw_evidence_avg"] = (
            float(np.mean(vals))
            if vals
            else np.nan
        )

        spring_wet = row.get("spring_dw_wet_prob_mean")
        late_wet = row.get("late_dw_wet_prob_mean")

        if (
            spring_wet is not None
            and late_wet is not None
            and spring_wet > 0
        ):
            row["dw_seasonal_persistence"] = late_wet / spring_wet
        else:
            row["dw_seasonal_persistence"] = np.nan

        output.append(row)

    return output


def main():
    ee.Initialize(project=GEE_PROJECT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    layer = find_wetland_layer(GPKG)

    print(f"Using GPKG layer: {layer}")

    wetlands_gdf = gpd.read_file(
        GPKG,
        layer=layer,
    )

    print(f"Wetlands found: {len(wetlands_gdf)}")

    wetlands = polygons_to_ee(wetlands_gdf)

    all_rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        print(f"Processing Dynamic World {year}...")

        # Whole growing season
        annual_img = hydroperiod_metrics(
            wetlands.geometry(),
            f"{year}-04-20",
            f"{year}-09-21",
        )

        # WetScan spring window
        spring_img = hydroperiod_metrics(
            wetlands.geometry(),
            f"{year}-04-20",
            f"{year}-06-21",
        )

        # WetScan late-summer window
        late_img = hydroperiod_metrics(
            wetlands.geometry(),
            f"{year}-07-15",
            f"{year}-09-21",
        )

        annual_rows = reduce_metrics(
            annual_img,
            wetlands,
            "annual_",
        )

        spring_rows = reduce_metrics(
            spring_img,
            wetlands,
            "spring_",
        )

        late_rows = reduce_metrics(
            late_img,
            wetlands,
            "late_",
        )

        rows = merge_metric_rows(
            annual_rows,
            spring_rows,
            late_rows,
            year,
        )

        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    preferred = [
        "site",
        "wetland_id",
        "year",
        "annual_dw_water_prob_mean",
        "annual_dw_floodedveg_prob_mean",
        "annual_dw_wet_prob_mean",
        "dw_evidence_avg",
        "annual_dw_water_label_freq",
        "annual_dw_floodedveg_label_freq",
        "annual_dw_n_obs",
        "spring_dw_wet_prob_mean",
        "late_dw_wet_prob_mean",
        "dw_seasonal_persistence",
    ]

    remaining = [
        c for c in df.columns
        if c not in preferred
    ]

    df = df[
        [c for c in preferred if c in df.columns]
        + remaining
    ]

    df = df.sort_values(
        ["wetland_id", "year"]
    )

    df.to_csv(
        OUT_CSV,
        index=False,
    )

    print()
    print(f"Created: {OUT_CSV}")
    print(f"Rows: {len(df)}")
    print()
    print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()