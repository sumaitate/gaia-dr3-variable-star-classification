import numpy as np
import pandas as pd

from gaia_variable_stars.features import build, get_period


def sine(period, n=80, seed=0):
    rng = np.random.default_rng(seed)
    time = np.sort(rng.uniform(0, 200, n))
    mag = 15 + 0.3 * np.sin(2 * np.pi * time / period) + rng.normal(0, 0.01, n)
    return time, mag


def test_period():
    time, mag = sine(0.5)
    period, power = get_period(time, mag)

    assert abs(period - 0.5) < 0.005
    assert power > 0.8


def test_build():
    stars = []

    for source_id, n in [(1, 60), (2, 5)]:
        time, mag = sine(0.5, n=n, seed=source_id)
        stars.append(
            pd.DataFrame(
                {
                    "source_id": source_id,
                    "class_name": "RR Lyrae",
                    "g_transit_time": time,
                    "g_transit_mag": mag,
                    "bp_rp_color": 0.8,
                }
            )
        )

    feat = build(pd.concat(stars), min_points=20)

    assert list(feat["source_id"]) == [1]
    assert np.isclose(feat.loc[0, "log_period"], np.log10(feat.loc[0, "period"]))