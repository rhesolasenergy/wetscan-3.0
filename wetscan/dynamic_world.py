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
    """Create Dynamic World hydroperiod metrics for one period.

    Bands returned:

    dw_water_prob_mean
        Mean Dynamic World open-water probability.

    dw_floodedveg_prob_mean
        Mean Dynamic World flooded-vegetation probability.

    dw_wet_prob_mean
        Mean of max(water, flooded vegetation) probability.

    dw_wet_frequency
        Fraction of valid observations where wet probability is
        greater than or equal to probability_threshold.

    dw_n_obs
        Number of valid Dynamic World observations per pixel.
    """
    import ee

    collection = dynamic_world_collection(region, start, end)

    water = collection.select("water")
    flooded = collection.select("flooded_vegetation")

    wet_probability = collection.map(_wet_probability)

    wet_frequency = wet_probability.map(
        lambda img: img.gte(probability_threshold).rename("wet")
    ).mean()

    return (
        ee.Image.cat(
            [
                water.mean().rename("dw_water_prob_mean"),
                flooded.mean().rename("dw_floodedveg_prob_mean"),
                wet_probability.mean().rename("dw_wet_prob_mean"),
                wet_frequency.rename("dw_wet_frequency"),
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