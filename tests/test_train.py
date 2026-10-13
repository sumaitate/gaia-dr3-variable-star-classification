import numpy as np
import pandas as pd

from gaia_variable_stars.config import CLASSES, FEATURES
from gaia_variable_stars.modeling.train import fit


def fake():
    rng = np.random.default_rng(0)
    rows = []

    for k, name in enumerate(CLASSES):
        for i in range(60):
            row = {f: rng.normal(k, 1) for f in FEATURES}
            row.update(source_id=k * 100 + i, class_name=name)
            rows.append(row)

    feat = pd.DataFrame(rows)
    feat.loc[::17, "bp_rp_color"] = np.nan

    return feat


def test_fit():
    out = fit(fake())

    assert out["f1"] > 0.8
    assert out["matrix"].shape == (4, 4)
    assert set(out["rank"]["feature"]) == set(FEATURES)