import numpy as np
import pandas as pd
from scipy.signal import lombscargle
from scipy.stats import kurtosis, skew
from tqdm import tqdm

from gaia_variable_stars.config import (
    CLEAN_FILE,
    FEATURE_FILE,
    MIN_POINTS,
)


STEP = 0.0002
FREQS = np.arange(0.001, 10.0, STEP)


def get_period(time, mag):
    mag = mag - mag.mean()

    power = lombscargle(time, mag, 2 * np.pi * FREQS, normalize=True)
    best = FREQS[power.argmax()]

    fine = np.linspace(best - STEP, best + STEP, 201)
    power = lombscargle(time, mag, 2 * np.pi * fine, normalize=True)

    return 1 / fine[power.argmax()], power.max()


def get_fourier(time, mag, period):
    phase = 2 * np.pi * time / period

    design = np.column_stack(
        [
            np.ones_like(time),
            np.sin(phase),
            np.cos(phase),
            np.sin(2 * phase),
            np.cos(2 * phase),
        ]
    )

    coef = np.linalg.lstsq(design, mag, rcond=None)[0]

    first = np.hypot(coef[1], coef[2])
    second = np.hypot(coef[3], coef[4])

    return first, second / first


def get_features(star):
    time = star["g_transit_time"].to_numpy()
    mag = star["g_transit_mag"].to_numpy()

    period, power = get_period(time, mag)
    fourier_amp, fourier_ratio = get_fourier(time, mag, period)
    low, high = np.percentile(mag, [5, 95])

    return {
        "magnitude_mean": mag.mean(),
        "magnitude_std": mag.std(ddof=1),
        "magnitude_skew": skew(mag),
        "magnitude_kurt": kurtosis(mag),
        "amplitude": high - low,
        "period": period,
        "log_period": np.log10(period),
        "power": power,
        "fourier_amp": fourier_amp,
        "fourier_ratio": fourier_ratio,
        "bp_rp_color": star["bp_rp_color"].iloc[0],
    }


def build(data, min_points=MIN_POINTS):
    """One feature row per star. Stars with too few epochs are skipped."""
    rows = []

    for source_id, star in tqdm(data.groupby("source_id")):
        if len(star) < min_points:
            continue

        row = {
            "source_id": source_id,
            "class_name": star["class_name"].iloc[0],
        }
        row.update(get_features(star))
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    feat = build(pd.read_csv(CLEAN_FILE, low_memory=False))

    FEATURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    feat.to_csv(FEATURE_FILE, index=False)


if __name__ == "__main__":
    main()