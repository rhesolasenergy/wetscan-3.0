# Changelog

## 0.2.0 - 2026-09-24
- Fixed: lush cropland and pasture were mapped as wetland (the v0.1 wet-vegetation
  rule was an absolute NDMI > 0.2 test, i.e. "green vegetation"). Wet vegetation
  now requires a local moisture anomaly vs the surrounding 500 m, persistence in
  >= 50 % of years, proximity to open water (or WorldCover water/wetland), and
  excludes cropland, built-up and slopes > 5 deg. Boreal peatland exception added.
- Changed: Earth Engine now exports index composites (int16) instead of masks so
  thresholds can be re-tuned locally with --skip-download. Old masks/ caches are
  incompatible (delete or --overwrite). Default analysis scale is now 20 m.
- Changed: open-water MNDWI threshold 0.05 -> 0.0 to catch mixed pixels in small potholes.
- Added: masks/years_failed.json and a report banner listing skipped years.
- Fixed: peatland class no longer assigned to basins that flood in spring.
- Tests: synthetic site now includes a lush cropland block and a uniformly moist
  pasture strip that must not be mapped.
