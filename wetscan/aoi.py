"""Load a project area of interest from KMZ / KML / GeoJSON / Shapefile / GeoPackage.

Returns a GeoDataFrame in EPSG:4326 with one row per AOI polygon and a
``name`` column (taken from the KML placemark name where available).
"""
from __future__ import annotations

import io
import math
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
from shapely.geometry import MultiPolygon, Polygon, shape
from shapely.ops import unary_union

KML_NS = {"kml": "http://www.opengis.net/kml/2.2"}


def _parse_coords(text: str) -> list[tuple[float, float]]:
    pts = []
    for tok in text.replace("\n", " ").split():
        parts = tok.split(",")
        if len(parts) >= 2:
            pts.append((float(parts[0]), float(parts[1])))
    return pts


def _polygons_from_kml(kml_bytes: bytes) -> list[dict]:
    """Minimal KML parser: Placemark -> Polygon / MultiGeometry of Polygons.

    Written by hand rather than relying on GDAL's KML driver so that it works
    on any install (the LIBKML driver is frequently missing on Windows/conda).
    """
    root = ET.fromstring(kml_bytes)
    # tolerate KML files with the older namespace or none at all
    ns_match = re.match(r"\{(.*)\}", root.tag)
    ns = {"kml": ns_match.group(1)} if ns_match else {}
    tag = (lambda t: f"kml:{t}") if ns else (lambda t: t)

    folder_tag = f"{{{ns['kml']}}}Folder" if ns else "Folder"
    pm_tag = f"{{{ns['kml']}}}Placemark" if ns else "Placemark"
    poly_tag = f"{{{ns['kml']}}}Polygon" if ns else "Polygon"

    out = []

    def _visit(el, folder: str):
        for ch in el:
            if ch.tag == folder_tag:
                nm = ch.find(tag("name"), ns)
                _visit(ch, (nm.text or "").strip() if nm is not None else folder)
            elif ch.tag == pm_tag:
                name_el = ch.find(tag("name"), ns)
                name = (name_el.text or "").strip() if name_el is not None else ""
                desc_el = ch.find(tag("description"), ns)
                desc = (desc_el.text or "").strip() if desc_el is not None else ""
                polys = []
                for poly in ch.iter(poly_tag):
                    outer = poly.find(f"{tag('outerBoundaryIs')}/{tag('LinearRing')}/{tag('coordinates')}", ns)
                    if outer is None or not outer.text:
                        continue
                    shell = _parse_coords(outer.text)
                    holes = []
                    for inner in poly.findall(f"{tag('innerBoundaryIs')}/{tag('LinearRing')}/{tag('coordinates')}", ns):
                        if inner.text:
                            holes.append(_parse_coords(inner.text))
                    if len(shell) >= 4:
                        polys.append(Polygon(shell, holes))
                if polys:
                    geom = polys[0] if len(polys) == 1 else MultiPolygon(polys)
                    out.append({"name": name, "folder": folder, "description": desc, "geometry": geom})
            else:
                _visit(ch, folder)

    _visit(root, "")
    return out


def load_aoi(path: str | Path, dissolve: bool = False) -> gpd.GeoDataFrame:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".kmz", ".kml"}:
        if suffix == ".kmz":
            with zipfile.ZipFile(path) as z:
                kml_names = [n for n in z.namelist() if n.lower().endswith(".kml")]
                if not kml_names:
                    raise ValueError(f"No .kml inside {path}")
                kml_bytes = z.read(kml_names[0])
        else:
            kml_bytes = path.read_bytes()
        recs = _polygons_from_kml(kml_bytes)
        if not recs:
            raise ValueError(f"No polygon placemarks found in {path}")
        gdf = gpd.GeoDataFrame(recs, geometry="geometry", crs="EPSG:4326")
    else:
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            raise ValueError(f"{path} has no CRS; please define one")
        gdf = gdf.to_crs("EPSG:4326")
        if "name" not in gdf.columns:
            for c in ("Name", "NAME", "site", "Site", "id", "ID"):
                if c in gdf.columns:
                    gdf["name"] = gdf[c].astype(str)
                    break
            else:
                gdf["name"] = [f"AOI_{i+1}" for i in range(len(gdf))]

    # keep polygons only
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf["name"] = [n if n else f"AOI_{i+1}" for i, n in enumerate(gdf["name"])]
    # make names filesystem-safe and unique
    seen: dict[str, int] = {}
    safe = []
    for n in gdf["name"]:
        s = re.sub(r"[^A-Za-z0-9_.-]+", "_", n).strip("_") or "AOI"
        seen[s] = seen.get(s, 0) + 1
        safe.append(s if seen[s] == 1 else f"{s}_{seen[s]}")
    gdf["slug"] = safe

    if dissolve:
        gdf = gpd.GeoDataFrame(
            {"name": ["AOI"], "slug": ["AOI"], "geometry": [unary_union(gdf.geometry)]},
            crs="EPSG:4326",
        )
    return gdf.reset_index(drop=True)


BUFFER_NAME_RE = re.compile(r"^\s*\d+\s*(m|km)\b", re.IGNORECASE)


def _site_slug(folder: str) -> str:
    # "Brighthold (Site 134)" -> "Brighthold_Site134"; "Polaris_Impact__Zones" -> "Polaris"
    s = re.sub(r"_?impact_+zones?", "", folder, flags=re.IGNORECASE)
    s = re.sub(r"\(\s*site\s*(\d+)\s*\)", r"Site\1", s, flags=re.IGNORECASE)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s or "Site"


def split_sites(gdf: gpd.GeoDataFrame, study_buffer_m: float = 2000.0) -> list[dict]:
    """Group AOI placemarks into project sites.

    Placemarks whose name starts with a distance ("800m Notification",
    "400m Consultation") are treated as *reference rings*; everything else in
    the same KML folder is a parcel of the *project area*. Files without
    folders are treated as one site.

    Returns one dict per site:
        name, slug, project_area (Polygon/MultiPolygon, EPSG:4326),
        rings {name: geometry}, study_area (project area buffered by
        ``study_buffer_m``), parcels (GeoDataFrame)
    """
    if "folder" not in gdf.columns or gdf["folder"].fillna("").eq("").all():
        gdf = gdf.assign(folder="Project")
    sites = []
    for folder, grp in gdf.groupby("folder", sort=False):
        is_ring = grp["name"].str.match(BUFFER_NAME_RE)
        parcels = grp[~is_ring]
        rings = grp[is_ring]
        if parcels.empty:  # only rings supplied: use the smallest ring as the project area
            parcels = rings.iloc[[rings.geometry.to_crs(utm_crs_for(rings)).area.argmin()]]
            rings = rings.drop(parcels.index)
        crs = utm_crs_for(parcels)
        project = unary_union(parcels.to_crs(crs).geometry.buffer(0.5).buffer(-0.5))  # heal slivers
        study = gpd.GeoSeries([project.buffer(study_buffer_m)], crs=crs).to_crs("EPSG:4326").iloc[0]
        project_wgs = gpd.GeoSeries([project], crs=crs).to_crs("EPSG:4326").iloc[0]
        sites.append({
            "name": folder,
            "slug": _site_slug(folder),
            "crs": crs,
            "project_area": project_wgs,
            "project_area_ha": project.area / 1e4,
            "rings": {n: g for n, g in zip(rings["name"], rings.geometry)},
            "study_area": study,
            "study_buffer_m": study_buffer_m,
            "parcels": parcels.reset_index(drop=True),
        })
    return sites


def utm_crs_for(gdf: gpd.GeoDataFrame) -> str:
    """EPSG code of the UTM zone containing the AOI centroid (northern hemisphere)."""
    c = gdf.to_crs("EPSG:4326").geometry.union_all().centroid
    zone = int(math.floor((c.x + 180) / 6) + 1)
    return f"EPSG:{32600 + zone if c.y >= 0 else 32700 + zone}"


def buffered_bounds(gdf: gpd.GeoDataFrame, buffer_m: float = 250.0) -> tuple[float, float, float, float]:
    """WGS84 bounds of the AOI buffered by ``buffer_m`` metres (so that wetlands
    straddling the boundary are captured whole)."""
    crs = utm_crs_for(gdf)
    return tuple(gdf.to_crs(crs).buffer(buffer_m).to_crs("EPSG:4326").total_bounds)
