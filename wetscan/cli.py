"""Command line interface.

    wetscan run  AOI.kmz --out results/                 full pipeline, every site in the file
    wetscan sites AOI.kmz                                list the sites/parcels found in a file
    wetscan precip --lon -112.2 --lat 51.47              precipitation classification only
    wetscan basemap AOI.kmz --provider google --out results/
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from . import __version__
from .aoi import load_aoi, split_sites
from .config import Config

app = typer.Typer(add_completion=False, help="Wetland extent and AWCS screening from multi-decade satellite imagery.")


def _setup_logging(verbose: bool):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)

@app.command("dynamic-world")
def dynamic_world(
    polygons: Path = typer.Argument(
        ...,
        help="WetScan wetland polygons: GPKG/SHP/GeoJSON",
    ),
    site: str = typer.Option(
        ...,
        help="site name / slug used in output files",
    ),
    out: Path = typer.Option(
        Path("dynamic_world_out"),
        help="output folder",
    ),
    start_year: int = typer.Option(
        2016,
        help="first Dynamic World analysis year",
    ),
    end_year: int = typer.Option(
        None,
        help="last year (default: current year)",
    ),
    spring_start: str = typer.Option(
        "04-20",
        help="spring window start MM-DD",
    ),
    spring_end: str = typer.Option(
        "06-20",
        help="spring window end MM-DD",
    ),
    late_start: str = typer.Option(
        "07-15",
        help="late-season window start MM-DD",
    ),
    late_end: str = typer.Option(
        "09-20",
        help="late-season window end MM-DD",
    ),
    scale_m: int = typer.Option(
        10,
        help="Dynamic World analysis scale in metres",
    ),
    layer: str = typer.Option(
        None,
        help="polygon layer name for GeoPackage inputs",
    ),
    gee_project: str = typer.Option(
        None,
        help="Earth Engine cloud project",
    ),
):
    """Build Dynamic World hydroperiod evidence tables for wetland polygons."""
    import pandas as pd

    from .dynamic_world import build_evidence_tables

    cfg = Config()

    project = (
        gee_project
        or cfg.keys.gee_project
    )

    if not project:
        raise typer.BadParameter(
            "Earth Engine project required via "
            "--gee-project or GEE_PROJECT."
        )

    final_year = (
        end_year
        or pd.Timestamp.today().year
    )

    result = build_evidence_tables(
        polygons_path=polygons,
        site=site,
        out_dir=out,
        start_year=start_year,
        end_year=final_year,
        gee_project=project,
        scale_m=scale_m,
        spring_start=spring_start,
        spring_end=spring_end,
        late_start=late_start,
        late_end=late_end,
        layer=layer,
    )

    typer.echo(
        json.dumps(
            result,
            indent=2,
        )
    )

@app.command()
def sites(aoi: Path, study_buffer_m: float = 2000.0):
    """List the project sites, parcels and reference rings found in an AOI file."""
    g = load_aoi(aoi)
    for s in split_sites(g, study_buffer_m):
        c = s["project_area"].centroid
        typer.echo(f"{s['slug']:24s} {s['crs']}  project {s['project_area_ha']:8.1f} ha  "
                   f"{len(s['parcels']):3d} parcels  rings={list(s['rings'])}  centroid {c.y:.4f},{c.x:.4f}")


@app.command()
def run(
    aoi: Path = typer.Argument(..., help="KMZ/KML/GeoJSON/SHP/GPKG with the project area(s)"),
    out: Path = typer.Option(Path("wetscan_out"), help="output folder"),
    site: list[str] = typer.Option(None, help="only run these site slugs (see `wetscan sites`)"),
    study_buffer_m: float = typer.Option(2000.0, help="analysis distance beyond the project area"),
    start_year: int = typer.Option(1972, help="first imagery year (1972 = Landsat-1 MSS)"),
    end_year: int = typer.Option(None, help="last imagery year (default: current year)"),
    scale_m: int = typer.Option(20, help="analysis pixel size in metres (10 = Sentinel-2 native, 4x download)"),
    precip_source: str = typer.Option("era5", help="era5 (Earth Engine) or eccc (station API, no GEE)"),
    natural_region: str = typer.Option(None, help="grassland|parkland|boreal|foothills (default: inferred from latitude)"),
    basemap: list[str] = typer.Option(None, help="google, azure and/or bing high-res mosaics for QA"),
    basemap_zoom: int = typer.Option(None, help="XYZ zoom for basemaps (default ~1 m/px)"),
    min_years: int = typer.Option(2, help="pixel must be wet in >= N years to enter the greatest extent"),
    min_area_ha: float = typer.Option(0.05, help="drop wetlands smaller than this"),
    include_mss_veg: bool = typer.Option(False, help="let 1972-83 MSS wet-vegetation masks contribute to extent"),
    skip_download: bool = typer.Option(False, help="re-use cached masks (re-run classification only)"),
    overwrite: bool = typer.Option(False, help="re-download everything"),
    gee_project: str = typer.Option(None, help="Earth Engine cloud project (or set GEE_PROJECT)"),
    verbose: bool = False,
):
    """Run the full pipeline for every site in the AOI file."""
    from . import imagery, pipeline
    _setup_logging(verbose)
    cfg = Config()
    cfg.start_year, cfg.end_year, cfg.scale_m = start_year, end_year, scale_m
    cfg.thresholds.min_years_for_extent = min_years
    cfg.thresholds.min_wetland_area_m2 = min_area_ha * 1e4
    if gee_project:
        cfg.keys.gee_project = gee_project
    if not skip_download or precip_source == "era5":
        imagery.init_ee(cfg.keys.gee_project)

    g = load_aoi(aoi)
    all_sites = split_sites(g, study_buffer_m)
    if site:
        all_sites = [s for s in all_sites if s["slug"] in site]
        if not all_sites:
            raise typer.BadParameter(f"no site matched {site}")
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for s in all_sites:
        try:
            results.append(pipeline.run_site(s, cfg, out, precip_source, natural_region, tuple(basemap or ()),
                                             basemap_zoom, overwrite, include_mss_veg, skip_download))
        except Exception as e:  # noqa: BLE001
            logging.exception("site %s failed: %s", s["slug"], e)
            results.append({"slug": s["slug"], "error": str(e)})
    (out / "run_summary.json").write_text(json.dumps(results, indent=2))
    for r in results:
        typer.echo(json.dumps(r))


@app.command()
def precip(lon: float, lat: float, start_year: int = 1972, end_year: int = None,
           source: str = "era5", station: str = None, out: Path = None, gee_project: str = None):
    """Classify every water year as wet/normal/dry for a point."""
    import pandas as pd
    from . import imagery, precip as pr
    cfg = Config()
    if source == "era5":
        imagery.init_ee(gee_project or cfg.keys.gee_project)
    res = pr.get_precip(lon, lat, start_year, end_year or pd.Timestamp.today().year, source, cfg.precip, station)
    typer.echo(res.years.to_string())
    if out:
        res.years.to_csv(out)


@app.command()
def basemap(aoi: Path, provider: str = "google", out: Path = Path("wetscan_out"), zoom: int = None,
            study_buffer_m: float = 2000.0):
    """Download a high-resolution Google / Azure (Bing) mosaic for each site."""
    from . import tiles
    from .aoi import buffered_bounds
    import geopandas as gpd
    _setup_logging(False)
    cfg = Config()
    prov = tiles.make_provider(provider, cfg.keys)
    for s in split_sites(load_aoi(aoi), study_buffer_m):
        b = buffered_bounds(gpd.GeoDataFrame(geometry=[s["study_area"]], crs="EPSG:4326"), 100)
        z = zoom or tiles.zoom_for_resolution(s["project_area"].centroid.y, 1.0, prov.px)
        p = tiles.mosaic(prov, b, z, out / s["slug"] / "basemaps" / f"{provider}_z{z}.tif")
        typer.echo(f"{s['slug']}: {p}")


@app.command()
def version():
    typer.echo(__version__)


if __name__ == "__main__":
    app()
