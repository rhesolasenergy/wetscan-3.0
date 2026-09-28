"""Per-site HTML report: precipitation history, extent by year, numbered
wetland map and the classification table. Charts are matplotlib PNGs embedded
as data URIs so the report is a single self-contained file."""
from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from jinja2 import Template  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# diverging: dry (orange) - neutral - wet (blue); categorical for AWCS classes
CLS_COL = {"very_dry": "#b8471f", "dry": "#eb6834", "normal": "#b9b8b1", "wet": "#2a78d6",
           "very_wet": "#174f8f", "no_data": "#e6e5e0"}
AWCS_COL = {"Shallow Open Water": "#2a78d6", "Marsh": "#1baf7a", "Swamp": "#eb6834",
            "Fen (bog possible)": "#4a3aa7", "Bog/Fen (peatland)": "#4a3aa7"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3de"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def precip_chart(pr: pd.DataFrame, years: list[int]) -> str:
    d = pr.reindex(years)
    fig, ax = plt.subplots(figsize=(11, 3.4))
    cols = [CLS_COL.get(c, CLS_COL["no_data"]) for c in d["wet_dry"].fillna("no_data")]
    ax.bar(d.index, d["precip_mm"].fillna(0), color=cols, width=0.8, linewidth=0)
    normal = pr["precip_mm"][(pr.index >= 1991) & (pr.index <= 2020)].mean()
    ax.axhline(normal, color=INK2, linewidth=1, linestyle="--")
    ax.text(d.index.min(), normal, f" 1991-2020 mean {normal:.0f} mm", va="bottom", fontsize=8, color=INK2)
    ax.set_ylabel("Water-year precipitation (mm)", color=INK2, fontsize=9)
    ax.set_title("Precipitation by water year (Oct-Sep), classified vs 1991-2020 normal", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.legend(handles=[Patch(color=CLS_COL[k], label=k.replace("_", " ")) for k in ("very_dry", "dry", "normal", "wet", "very_wet")],
              ncol=5, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12))
    return _png(fig)


def extent_chart(yearly: pd.DataFrame) -> str:
    d = yearly.copy()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ok = d["coverage_pct"] >= 50
    ax.plot(d.index[ok], d.loc[ok, "wet_total_ha"], color="#2a78d6", linewidth=2, label="water + wet vegetation")
    ax.plot(d.index[ok], d.loc[ok, "late_water_ha"], color="#174f8f", linewidth=2, linestyle=":", label="late-summer open water")
    ax.scatter(d.index[ok], d.loc[ok, "spring_water_ha"], s=18, color="#1baf7a", label="spring open water", zorder=3)
    if (~ok).any():
        ax.scatter(d.index[~ok], np.zeros((~ok).sum()), marker="x", color=INK2, s=20, label="< 50 % coverage (no/cloudy scenes)")
    mss = d["sensor"].eq("MSS")
    if mss.any():
        ax.axvspan(d.index[mss].min() - 0.5, d.index[mss].max() + 0.5, color="#f3f2ee", zorder=0)
        ax.text(d.index[mss].min(), ax.get_ylim()[1] * 0.95, "Landsat MSS (60 m)", fontsize=8, color=INK2, va="top")
    ax.set_ylabel("Area inside greatest extent (ha)", color=INK2, fontsize=9)
    ax.set_title("Wet area by year", loc="left", fontsize=11, color=INK)
    _style(ax)
    ax.legend(ncol=4, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12))
    return _png(fig)


def map_figure(ctx) -> str:
    import geopandas as gpd
    from rasterio.plot import plotting_extent
    st, w = ctx["stack"], ctx["wetlands"]
    ext = plotting_extent(ctx["freq"]["f_wet"], st.transform)
    fig, ax = plt.subplots(figsize=(10, 10))
    f = ctx["freq"]["f_wet"]
    ax.imshow(np.where(np.isnan(f), 0, f), extent=ext, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    pa = gpd.GeoSeries([ctx["site"]["project_area"]], crs="EPSG:4326").to_crs(st.crs)
    pa.boundary.plot(ax=ax, color="#e34948", linewidth=1.5)
    for n, g in ctx["site"]["rings"].items():
        gpd.GeoSeries([g], crs="EPSG:4326").to_crs(st.crs).boundary.plot(ax=ax, color="#e34948", linewidth=0.7, linestyle="--")
    if not w.empty:
        w.boundary.plot(ax=ax, color=[AWCS_COL.get(c, "#000") for c in w["awcs_class"]], linewidth=1.2)
        for _, r in w.iterrows():
            c = r.geometry.representative_point()
            ax.annotate(r.wetland_id.split("-")[-1], (c.x, c.y), fontsize=6.5, ha="center", va="center", color=INK,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.set_title(f"{ctx['site']['name']}: inundation frequency 1972-present with numbered wetlands", loc="left", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(handles=[Patch(facecolor="none", edgecolor=v, label=k) for k, v in AWCS_COL.items() if k != "Bog/Fen (peatland)"]
              + [Patch(facecolor="none", edgecolor="#e34948", label="project area / rings")],
              loc="lower left", fontsize=8, frameon=True)
    return _png(fig)


TEMPLATE = Template("""<!doctype html><html><head><meta charset="utf-8">
<title>{{ site.name }} wetland screening</title>
<style>
body{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#0b0b0b;max-width:1180px;margin:24px auto;padding:0 16px;line-height:1.45}
h1{font-size:22px;margin-bottom:2px} h2{font-size:16px;margin-top:28px;border-bottom:1px solid #e4e3de;padding-bottom:4px}
.sub{color:#52514e;font-size:13px} table{border-collapse:collapse;font-size:12px;width:100%} th,td{padding:4px 6px;border-bottom:1px solid #eee;text-align:left;vertical-align:top}
th{background:#f6f5f2;position:sticky;top:0} .tiles{display:flex;gap:14px;flex-wrap:wrap;margin:14px 0}
.tile{background:#f6f5f2;border-radius:8px;padding:10px 14px;min-width:150px} .tile b{display:block;font-size:20px} .tile span{font-size:12px;color:#52514e}
img{max-width:100%} .note{background:#fff7e6;border-left:3px solid #eda100;padding:8px 12px;font-size:13px}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;color:#fff}
</style></head><body>
<h1>{{ site.name }} - wetland extent &amp; AWCS screening</h1>
<div class="sub">Project area {{ '%.1f'|format(site.project_area_ha) }} ha; study area = project area + {{ site.study_buffer_m|int }} m.
Natural region assumed: <b>{{ region }}</b>. Imagery years {{ years[0] }}-{{ years[-1] }} ({{ years|length }} yrs). Precipitation: {{ precip_meta.dataset }}.</div>

<div class="tiles">
 <div class="tile"><b>{{ wetlands|length }}</b><span>wetlands numbered</span></div>
 <div class="tile"><b>{{ n_project }}</b><span>touch the project area</span></div>
 <div class="tile"><b>{{ greatest_extent_ha }} ha</b><span>greatest extent, all years</span></div>
 <div class="tile"><b>{{ wet_year_extent_ha }} ha</b><span>extent in wet years ({{ wet_years|length }} yrs)</span></div>
 <div class="tile"><b>{{ dry_year_extent_ha }} ha</b><span>extent in dry years ({{ dry_years|length }} yrs)</span></div>
</div>

{% if years_failed %}<div class="note">Years with no usable imagery or a download error (not in the analysis): {{ years_failed.keys()|join(', ') }}. See masks/years_failed.json.</div>{% endif %}
<div class="note">Desktop screening only. AWCS classes are estimated from 10-60 m optical imagery and land-cover priors; bog vs fen,
water depth and peat presence cannot be determined remotely and require field verification under the Alberta Wetland Identification
and Delineation Directive. Landsat MSS years (1972-83) are 60 m and contribute open water only.
Rules: open water = MNDWI &gt; {{ config.thresholds.mndwi_water }}; wet vegetation = late-summer NDMI at least {{ config.thresholds.ndmi_anomaly }} above the surrounding {{ config.thresholds.wetveg_window_m|int }} m mean, NDMI &gt; {{ config.thresholds.ndmi_wet }}, NDVI &gt; {{ config.thresholds.ndvi_veg_min }}, in &ge; {{ (100*config.thresholds.wetveg_min_persistence)|int }}% of years, within {{ config.thresholds.water_proximity_m|int }} m of open water (any year) or WorldCover water/wetland; cropland, built-up and slopes &gt; {{ config.thresholds.max_slope_deg }}&deg; excluded from the vegetation rule.</div>

<h2>Precipitation history</h2>
<img src="{{ precip_png }}">
<p class="sub">Wet years used for the wet-year extent: {{ wet_years|join(', ') or 'none' }}. Dry years: {{ dry_years|join(', ') or 'none' }}.
Classification uses the 24-month antecedent total where available (wetland levels lag rainfall), otherwise the single water year.</p>

<h2>Wet area by year</h2>
<img src="{{ extent_png }}">

<h2>Map</h2>
<img src="{{ map_png }}">
{% if basemaps %}<p class="sub">High-resolution basemap mosaics saved for QA: {% for k, v in basemaps.items() %}{{ k }} ({{ v.name }}) {% endfor %}</p>{% endif %}

<h2>Wetlands</h2>
<table><thead><tr><th>ID</th><th>Location</th><th>Area (ha)</th><th>Wet-yr ha</th><th>Dry-yr ha</th><th>AWCS class</th><th>Form</th><th>S&amp;K</th>
<th>Spring water %</th><th>Late water %</th><th>Wet veg %</th><th>Woody %</th><th>Conf.</th><th>Rationale</th></tr></thead><tbody>
{% for r in rows %}<tr><td><b>{{ r.wetland_id }}</b></td><td>{{ r.location }}</td><td>{{ '%.2f'|format(r.area_ha) }}</td>
<td>{{ '%.2f'|format(r.wet_year_extent_ha) }}</td><td>{{ '%.2f'|format(r.dry_year_extent_ha) }}</td>
<td><span class="pill" style="background:{{ awcs_col.get(r.awcs_class,'#333') }}">{{ r.awcs_class }}</span></td><td>{{ r.awcs_form }}</td>
<td>{{ r.sk_class }} {{ r.permanence }}</td><td>{{ '%.0f'|format(100*(r.f_spring or 0)) }}</td><td>{{ '%.0f'|format(100*(r.f_late or 0)) }}</td>
<td>{{ '%.0f'|format(100*(r.f_wetveg or 0)) }}</td><td>{{ '%.0f'|format(100*((r.frac_tree or 0)+(r.frac_shrub or 0))) }}</td>
<td>{{ r.confidence }}</td><td class="sub">{{ r.rationale }}</td></tr>{% endfor %}
</tbody></table>

<h2>Summary by class and location</h2>
<table><thead><tr><th>Location</th><th>AWCS class</th><th>Count</th><th>Area (ha)</th></tr></thead><tbody>
{% for r in summary %}<tr><td>{{ r.location }}</td><td>{{ r.awcs_class }}</td><td>{{ r.count }}</td><td>{{ r.area_ha }}</td></tr>{% endfor %}
</tbody></table>

<h2>Year table</h2>
<table><thead><tr><th>Year</th><th>Sensor</th><th>Coverage %</th><th>Spring water ha</th><th>Late water ha</th><th>Wet veg ha</th><th>Wet total ha</th>
<th>Precip mm</th><th>% of normal</th><th>Class</th><th>24-mo class</th><th>SPI-12</th></tr></thead><tbody>
{% for y, r in yearly %}<tr><td>{{ y }}</td><td>{{ r.sensor }}</td><td>{{ '%.0f'|format(r.coverage_pct) }}</td><td>{{ r.spring_water_ha }}</td>
<td>{{ r.late_water_ha }}</td><td>{{ r.wet_veg_ha }}</td><td>{{ r.wet_total_ha }}</td><td>{{ r.precip_mm }}</td><td>{{ r.pct_of_normal }}</td>
<td>{{ r.wet_dry }}</td><td>{{ r.wet_dry_24mo }}</td><td>{{ r.spi12 }}</td></tr>{% endfor %}
</tbody></table>
<p class="sub">Generated by wetscan {{ version }}. Thresholds: {{ config.thresholds }}</p>
</body></html>""")


def render(ctx: dict, path: Path) -> Path:
    from . import __version__
    w = ctx["wetlands"]
    rows = w.drop(columns="geometry").fillna(np.nan).replace({np.nan: None}).to_dict("records") if not w.empty else []
    yearly = ctx["yearly"].replace({np.nan: None})
    html = TEMPLATE.render(
        **{k: v for k, v in ctx.items() if k not in ("stack", "extent_mask", "freq", "summary", "yearly")},
        rows=rows, n_project=int((w["location"] == "project_area").sum()) if not w.empty else 0,
        summary=ctx["summary"].to_dict("records"), yearly=[(y, r) for y, r in yearly.iterrows()],
        precip_png=precip_chart(ctx["precip"], ctx["years"]), extent_png=extent_chart(ctx["yearly"]),
        map_png=map_figure(ctx), awcs_col=AWCS_COL, version=__version__,
    )
    path.write_text(html, encoding="utf-8")
    return path
