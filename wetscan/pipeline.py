"""End-to-end run for one project site (used by the CLI)."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio

from . import classify, extent, imagery, precip, report, tiles
from .aoi import buffered_bounds
from .config import Config

log = logging.getLogger("wetscan.pipeline")


def run_site(site: dict, cfg: Config, out_root: Path, precip_source: str = "era5",
             natural_region: str | None = None, basemaps: tuple[str, ...] = (),
             basemap_zoom: int | None = None, overwrite: bool = False,
             include_mss_veg: bool = False, skip_download: bool = False) -> dict:
    slug = site["slug"]
    out = out_root / slug
    out.mkdir(parents=True, exist_ok=True)
    crs = cfg.working_crs or site["crs"]
    study = gpd.GeoDataFrame(geometry=[site["study_area"]], crs="EPSG:4326")
    bounds = tuple(float(b) for b in buffered_bounds(study, 100))
    c = site["project_area"].centroid
    region = natural_region or classify.natural_region_from_lat(c.y)
    end_year = cfg.end_year or pd.Timestamp.today().year
    years = list(range(cfg.start_year, end_year + 1))
    log.info("site %s: %.0f ha project area, study bounds %s, region=%s", slug, site["project_area_ha"], bounds, region)

    # ---- 1. precipitation ------------------------------------------------ #
    pr_path = out / "precip_years.csv"
    if pr_path.exists() and not overwrite:
        pr_years = pd.read_csv(pr_path, index_col="water_year")
        pr_meta = json.loads((out / "precip_meta.json").read_text())
    else:
        pr = precip.get_precip(c.x, c.y, cfg.start_year, end_year, source=precip_source, cfg=cfg.precip)
        pr.years.to_csv(pr_path)
        pr.monthly.to_csv(out / "precip_monthly.csv", header=True)
        pr_meta = pr.meta
        (out / "precip_meta.json").write_text(json.dumps(pr_meta, indent=2, default=str))
        pr_years = pr.years
    year_class = pr_years["wet_dry_24mo"].where(pr_years["wet_dry_24mo"].ne("no_data"), pr_years["wet_dry"])

    # ---- 2. imagery masks ----------------------------------------------- #
    mask_dir = out / "masks"
    if skip_download:
        paths = {int(k): Path(v) for k, v in json.loads((mask_dir / "years.json").read_text()).items()}
        paths = {k: (v if v.exists() else mask_dir / v.name) for k, v in paths.items()}
    else:
        paths = imagery.fetch_year_stack(bounds, crs, cfg, mask_dir, years, overwrite=overwrite)
    thr = cfg.thresholds
    st = extent.load_stack(paths, mask_dir / "static_layers.tif", thr,
                           peat_region=region in classify.PEATLAND_REGIONS)

    # ---- 3. greatest extent --------------------------------------------- #
    freq = extent.frequency_layers(st, year_class, include_mss_veg)
    ge = extent.greatest_extent_mask(st, thr, include_mss_veg)
    wet_years = [y for y in st.years if year_class.get(y) in ("wet", "very_wet")]
    dry_years = [y for y in st.years if year_class.get(y) in ("dry", "very_dry")]
    ge_wet = extent.greatest_extent_mask(st, thr, include_mss_veg, wet_years) if len(wet_years) >= thr.min_years_for_extent else np.zeros_like(ge)
    ge_dry = extent.greatest_extent_mask(st, thr, include_mss_veg, dry_years) if len(dry_years) >= thr.min_years_for_extent else np.zeros_like(ge)
    _write_raster(out / "greatest_extent.tif", ge.astype("uint8"), st, nodata=None)
    _write_raster(out / "inundation_frequency.tif", np.nan_to_num(freq["f_wet"], nan=-1).astype("float32"), st, nodata=-1)

    # ---- 4. polygons, numbering, stats ---------------------------------- #
    w = extent.vectorise(ge, st.transform, st.crs, thr.smooth_buffer_m)
    prefix = slug[:3].upper()
    w = extent.number_wetlands(w, site["project_area"], site["rings"], prefix)
    layers = {k: v for k, v in freq.items() if k.startswith("f_")}
    layers["n_obs_years"] = freq["n_obs_years"].astype(float)
    for k in ("jrc_occurrence", "slope_deg", "treecover_pct"):
        if k in st.static:
            layers[k] = st.static[k].astype(float)
    w = extent.zonal_stats(w, layers, st.transform)
    w = extent.zonal_core_stats(w, freq, st.transform, thr.permanent_late_freq, thr.seasonal_spring_freq)
    if "worldcover" in st.static:
        w = extent.class_fraction(w, st.static["worldcover"].astype(int), st.transform, classify.WORLDCOVER)
    # extent in wet years vs dry years per wetland
    w = extent.zonal_stats(w, {"in_wet_year_extent": ge_wet.astype(float), "in_dry_year_extent": ge_dry.astype(float)}, st.transform)
    w["wet_year_extent_ha"] = (w["in_wet_year_extent"] * w["area_ha"]).round(2)
    w["dry_year_extent_ha"] = (w["in_dry_year_extent"] * w["area_ha"]).round(2)
    w["natural_region"] = region
    w["site"] = site["name"]
    w = classify.classify_wetlands(w, thr)
    w["area_ha"] = w["area_ha"].round(3)

    # ---- 5. yearly table ------------------------------------------------- #
    yearly = extent.yearly_extent_table(st, ge, pr_years, include_mss_veg)
    yearly.to_csv(out / "extent_by_year.csv")

    # ---- 6. outputs ------------------------------------------------------ #
    gpkg = out / f"{slug}_wetlands.gpkg"
    if gpkg.exists():
        gpkg.unlink()
    w.to_file(gpkg, layer="wetlands", driver="GPKG")
    gpd.GeoDataFrame({"name": [site["name"]]}, geometry=[site["project_area"]], crs="EPSG:4326").to_crs(st.crs).to_file(gpkg, layer="project_area", driver="GPKG")
    gpd.GeoDataFrame({"name": [f"study_area_{int(site['study_buffer_m'])}m"]}, geometry=[site["study_area"]], crs="EPSG:4326").to_crs(st.crs).to_file(gpkg, layer="study_area", driver="GPKG")
    if site["rings"]:
        gpd.GeoDataFrame({"name": list(site["rings"])}, geometry=list(site["rings"].values()), crs="EPSG:4326").to_crs(st.crs).to_file(gpkg, layer="reference_rings", driver="GPKG")
    w.drop(columns="geometry").to_csv(out / f"{slug}_wetlands.csv", index=False)
    w.to_crs("EPSG:4326").to_file(out / f"{slug}_wetlands.geojson", driver="GeoJSON")
    _write_kml(w, out / f"{slug}_wetlands.kml")

    # ---- 7. basemaps ----------------------------------------------------- #
    basemap_paths = {}
    for bm in basemaps:
        try:
            prov = tiles.make_provider(bm, cfg.keys)
            z = basemap_zoom or tiles.zoom_for_resolution(c.y, 1.0, prov.px)
            basemap_paths[bm] = tiles.mosaic(prov, bounds, z, out / "basemaps" / f"{bm}_z{z}.tif")
        except Exception as e:  # noqa: BLE001
            log.warning("basemap %s skipped: %s", bm, e)

    # ---- 8. report ------------------------------------------------------- #
    summary = classify.summarise(w)
    summary.to_csv(out / "summary_by_class.csv", index=False)
    ctx = {
        "site": site, "slug": slug, "region": region, "years": st.years, "precip": pr_years,
        "precip_meta": pr_meta, "yearly": yearly, "wetlands": w, "summary": summary,
        "greatest_extent_ha": round(float(ge.sum() * st.pixel_area_m2 / 1e4), 2),
        "wet_year_extent_ha": round(float(ge_wet.sum() * st.pixel_area_m2 / 1e4), 2),
        "dry_year_extent_ha": round(float(ge_dry.sum() * st.pixel_area_m2 / 1e4), 2),
        "wet_years": wet_years, "dry_years": dry_years, "basemaps": basemap_paths,
        "config": asdict(cfg), "stack": st, "extent_mask": ge, "freq": freq,
        "years_failed": json.loads((mask_dir / "years_failed.json").read_text()) if (mask_dir / "years_failed.json").exists() else {},
    }
    html = report.render(ctx, out / f"{slug}_report.html")
    log.info("site %s done: %d wetlands, greatest extent %.1f ha", slug, len(w), ctx["greatest_extent_ha"])
    return {"slug": slug, "n_wetlands": int(len(w)), "greatest_extent_ha": ctx["greatest_extent_ha"],
            "report": str(html), "gpkg": str(gpkg)}


def _write_raster(path: Path, arr: np.ndarray, st: extent.YearStack, nodata):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs=st.crs, transform=st.transform, nodata=nodata, compress="lzw") as dst:
        dst.write(arr, 1)


def _write_kml(w: gpd.GeoDataFrame, path: Path) -> None:
    """Minimal KML (Google Earth) with numbered, colour-coded wetlands."""
    colours = {"Shallow Open Water": "ffd67a2a", "Marsh": "ff7aaf1b", "Swamp": "ff3468eb",
               "Fen (bog possible)": "ffa43a4a", "Bog/Fen (peatland)": "ffa43a4a"}
    g = w.to_crs("EPSG:4326")
    parts = ['<?xml version="1.0" encoding="UTF-8"?>', '<kml xmlns="http://www.opengis.net/kml/2.2"><Document>',
             f"<name>{path.stem}</name>"]
    for cls, col in colours.items():
        sid = cls.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        parts.append(f'<Style id="{sid}"><LineStyle><color>{col}</color><width>2</width></LineStyle>'
                     f'<PolyStyle><color>66{col[2:]}</color></PolyStyle></Style>')
    for _, r in g.iterrows():
        sid = r.awcs_class.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        desc = (f"AWCS: {r.awcs_class} ({r.awcs_form})<br/>Permanence: S&amp;K {r.sk_class} {r.permanence}<br/>"
                f"Area: {r.area_ha:.2f} ha<br/>Location: {r.location}<br/>Confidence: {r.confidence}<br/>{r.rationale}")
        geoms = list(r.geometry.geoms) if r.geometry.geom_type == "MultiPolygon" else [r.geometry]
        polys = []
        for p in geoms:
            ring = " ".join(f"{x:.6f},{y:.6f},0" for x, y in p.exterior.coords)
            inner = "".join(f"<innerBoundaryIs><LinearRing><coordinates>{' '.join(f'{x:.6f},{y:.6f},0' for x, y in h.coords)}</coordinates></LinearRing></innerBoundaryIs>" for h in p.interiors)
            polys.append(f"<Polygon><outerBoundaryIs><LinearRing><coordinates>{ring}</coordinates></LinearRing></outerBoundaryIs>{inner}</Polygon>")
        geom_xml = polys[0] if len(polys) == 1 else f"<MultiGeometry>{''.join(polys)}</MultiGeometry>"
        parts.append(f'<Placemark><name>{r.wetland_id}</name><description><![CDATA[{desc}]]></description>'
                     f'<styleUrl>#{sid}</styleUrl>{geom_xml}</Placemark>')
    parts.append("</Document></kml>")
    path.write_text("\n".join(parts), encoding="utf-8")
