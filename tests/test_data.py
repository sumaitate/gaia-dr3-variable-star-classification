import numpy as np
import pandas as pd

from gaia_variable_stars.dataset import clean


def test_clean():
    data = pd.DataFrame(
        {
            "source_id": [1, 1, 1, 1, 1],
            "transit_id": [10, 11, 12, 13, 13],
            "g_transit_time": [1.0, 2.0, np.nan, 4.0, 4.0],
            "g_transit_mag": [15.0, 15.1, 15.2, 15.3, 15.3],
            "variability_flag_g_reject": ["False", "True", "False", "False", "False"],
        }
    )

    # row 2 rejected, row 3 has no time, last row duplicates transit 13
    assert list(clean(data)["transit_id"]) == [10, 13]