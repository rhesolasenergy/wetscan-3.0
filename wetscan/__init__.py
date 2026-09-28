"""wetscan - wetland extent and AWCS classification from multi-decade satellite imagery.

Pipeline (see cli.py):

    AOI (KMZ/KML/GeoJSON/SHP/GPKG)
      -> precip.py      : wet / normal / dry classification of every water-year
      -> imagery.py     : per-year spring & late-summer water and wet-vegetation
                          masks from Landsat MSS/TM/ETM+/OLI and Sentinel-2 (GEE)
      -> extent.py      : union across years = greatest extent; inundation
                          frequencies; per-wetland polygons
      -> classify.py    : Stewart & Kantrud permanence + AWCS class estimate
      -> tiles.py       : Google / Bing(Azure) high-resolution basemap mosaics
      -> report.py      : HTML report + GeoPackage + CSV outputs
"""

__version__ = "0.2.0"
