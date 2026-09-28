import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import synthetic  # noqa: E402

from wetscan import classify, extent, precip  # noqa: E402
from wetscan.config import Config, Thresholds  # noqa: E402


@pytest.fixture(scope="module")
def site_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("run") / "Synthetic_Site1"
    monthly = synthetic.synthetic_precip()
    years = precip.classify_years(monthly)
    out.mkdir(parents=True)
    years.to_csv(out / "precip_years.csv")
    (out / "precip_meta.json").write_text(json.dumps({"dataset": "synthetic"}))
    yc = years["wet_dry_24mo"].where(years["wet_dry_24mo"].ne("no_data"), years["wet_dry"])
    truth = synthetic.build(out, yc)
    return out, years, truth


def test_precip_classes_are_balanced():
    years = precip.classify_years(synthetic.synthetic_precip())
    full = years[years.n_months >= 12]
    counts = full["wet_dry"].value_counts()
    assert set(counts.index) <= set(precip.CLASS_ORDER)
    # by construction ~20 % dry-ish, ~20 % wet-ish within the normal period
    normal = full.loc[1991:2020]
    assert 3 <= normal["wet_dry"].isin(["dry", "very_dry"]).sum() <= 9
    assert 3 <= normal["wet_dry"].isin(["wet", "very_wet"]).sum() <= 9
    assert (full["n_months"] == 12).all()
    assert full["spi12"].abs().max() < 4


def test_spi_monotonic():
    s = synthetic.synthetic_precip()
    z = precip.spi(s, 12).dropna()
    roll = s.rolling(12).sum().reindex(z.index)
    assert np.corrcoef(z, roll)[0, 1] > 0.95


def test_stack_and_extent(site_dir):
    out, years, truth = site_dir
    paths = {int(k): Path(v) for k, v in json.loads((out / "masks" / "years.json").read_text()).items()}
    st = extent.load_stack(paths, out / "masks" / "static_layers.tif")
    assert st.w_spr.shape[0] == len(synthetic.YEARS)
    assert st.is_mss.sum() == 12
    # cloudy years contribute no observations
    i = st.years.index(1988)
    assert not st.obs_spr[i].any()
    ge = extent.greatest_extent_mask(st, Thresholds())
    # greatest extent covers the pond max, marsh max and the ephemeral depression
    for k in ("pond", "marsh", "ephem"):
        assert (ge & truth[k]).sum() / truth[k].sum() > 0.9, k
    # lush cropland and the uniformly moist pasture strip must NOT be wetland
    assert (ge & truth["field"]).sum() / truth["field"].sum() < 0.02, "cropland mapped as wetland"
    assert (ge & truth["pasture"]).sum() / truth["pasture"].sum() < 0.02, "pasture mapped as wetland"
    # non-boreal run: the treed block far from water is not kept (no peatland exception)
    assert (ge & truth["peat"]).sum() / truth["peat"].sum() < 0.1
    # boreal run: the persistently saturated treed block IS kept as a peatland candidate
    st_b = extent.load_stack(paths, out / "masks" / "static_layers.tif", Thresholds(), peat_region=True)
    ge_b = extent.greatest_extent_mask(st_b, Thresholds())
    assert (ge_b & truth["peat"]).sum() / truth["peat"].sum() > 0.9
    assert (ge_b & truth["field"]).sum() / truth["field"].sum() < 0.02
    assert ge_b.sum() < (truth["pond"] | truth["marsh"] | truth["peat"] | truth["ephem"]).sum() * 1.3


def test_end_to_end(site_dir):
    from wetscan import pipeline
    out, years, truth = site_dir
    cfg = Config()
    cfg.start_year, cfg.end_year = 1972, 2025
    site = synthetic.synthetic_site()
    site["slug"] = out.name
    res = pipeline.run_site(site, cfg, out.parent, skip_download=True)
    assert 3 <= res["n_wetlands"] <= 6
    import geopandas as gpd
    w = gpd.read_file(res["gpkg"], layer="wetlands")
    ids = list(w.wetland_id)
    assert ids == sorted(ids) and ids[0].endswith("W01")
    by_class = w.groupby("awcs_class")["area_ha"].sum()
    assert "Shallow Open Water" in by_class.index
    assert "Marsh" in by_class.index
    assert w.area_ha.max() < 30, "a polygon the size of the study area means the vegetation rule leaked"
    # the treed, never-flooded block in a boreal region is a peatland candidate
    site_boreal = dict(site)
    res2 = pipeline.run_site(site_boreal, cfg, out.parent, natural_region="boreal", skip_download=True)
    w2 = gpd.read_file(res2["gpkg"], layer="wetlands")
    assert w2["awcs_class"].str.startswith("Fen").any()
    # permanence: pond is S&K V, marsh core III or IV, ephemeral I-II
    sow = w2[w2.awcs_class == "Shallow Open Water"]
    assert (sow.sk_class == 5).all()
    assert (w2.sk_class <= 2).any()
    assert Path(res2["report"]).stat().st_size > 50_000
    assert (out / f"{out.name}_wetlands.kml").exists()
    assert (out / "extent_by_year.csv").exists()
    ye = pd.read_csv(out / "extent_by_year.csv", index_col="year")
    assert ye.loc[1988, "coverage_pct"] == 0
    wet = ye[ye.wet_dry.isin(["wet", "very_wet"])]["spring_water_ha"].mean()
    dry = ye[ye.wet_dry.isin(["dry", "very_dry"])]["spring_water_ha"].mean()
    assert wet > dry


def test_classifier_rules():
    thr = Thresholds()
    base = dict(f_spring=0.9, f_late=0.95, f_wetveg=0.0, f_wet=0.95, frac_water=0.8, jrc_occurrence=90,
                frac_tree=0, frac_shrub=0, frac_herb_wet=0, frac_crop=0, n_obs_years=40, slope_deg=1,
                natural_region="parkland")
    r = classify.classify_row(pd.Series(base), thr)
    assert r["awcs_class"] == "Shallow Open Water" and r["sk_class"] == 5
    r = classify.classify_row(pd.Series({**base, "f_late": 0.2, "f_spring": 0.8, "frac_water": 0, "jrc_occurrence": 5,
                                         "f_wetveg": 0.6}), thr)
    assert r["awcs_class"] == "Marsh" and r["sk_class"] == 3
    r = classify.classify_row(pd.Series({**base, "f_late": 0.05, "f_spring": 0.1, "f_wetveg": 0.9, "frac_tree": 0.7,
                                         "frac_water": 0, "jrc_occurrence": 0, "natural_region": "boreal",
                                         "f_late_wet_yrs": 0.05, "f_late_dry_yrs": 0.02}), thr)
    assert r["awcs_class"].startswith("Fen")
    r = classify.classify_row(pd.Series({**base, "f_late": 0.3, "f_spring": 0.7, "f_wetveg": 0.5, "frac_tree": 0.6,
                                         "frac_water": 0, "jrc_occurrence": 0}), thr)
    assert r["awcs_class"] == "Swamp"
