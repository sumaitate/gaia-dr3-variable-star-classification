import time
import numpy as np
import pandas as pd
from astroquery.gaia import Gaia

from gaia_variable_stars.config import (
    BATCH_ROOT,
    CLEAN_FILE,
    LABELED_FILE,
    PHOTOMETRY_FILE,
    RAW_ROOT,
    SOURCE_FILE,
)

SAMPLE_SIZE = 500
BATCH_SIZE = 10

CLASS_TABLES = {
    "RR Lyrae": "vari_rrlyrae",
    "Cepheid": "vari_cepheid",
    "Eclipsing Binary": "vari_eclipsing_binary",
    "Long-Period Variable": "vari_long_period_variable",
}

PERIOD_COLUMNS = {
    "RR Lyrae": "variability.pf AS gaia_pf, variability.p1_o AS gaia_p1_o",
    "Cepheid": "variability.pf AS gaia_pf, variability.p1_o AS gaia_p1_o, variability.p2_o AS gaia_p2_o",
    "Eclipsing Binary": "variability.frequency AS gaia_frequency",
    "Long-Period Variable": "variability.frequency AS gaia_frequency",
}

PERIOD_FILTERS = {
    "RR Lyrae": "(variability.pf > 0 OR variability.p1_o > 0)",
    "Cepheid": "(variability.pf > 0 OR variability.p1_o > 0 OR variability.p2_o > 0)",
    "Eclipsing Binary": "variability.frequency > 0",
    "Long-Period Variable": "variability.frequency > 0",
}


def normalize(data):
    data = data.copy()

    if "band" in data.columns:
        data = data[
            data["band"].astype("string").str.upper().eq("G")
        ].copy()

    names = {
        "time": "g_transit_time",
        "mag": "g_transit_mag",
        "flux": "g_transit_flux",
        "flux_error": "g_transit_flux_error",
        "n_obs": "g_transit_n_obs",
        "rejected_by_variability": "variability_flag_g_reject",
        "rejected_by_photometry": "photometry_flag_g_reject",
    }

    data = data.rename(columns={
        old: new
        for old, new in names.items()
        if old in data.columns and new not in data.columns
    })

    if (
        "photometry_flag_noisy_data" not in data.columns
        and "other_flags" in data.columns
    ):
        flags = pd.to_numeric(
            data["other_flags"], errors="coerce"
        ).astype("Int64")

        # Gaia DR3 other_flags: bit 13 identifies noisy G photometry.
        data["photometry_flag_noisy_data"] = (
            flags.fillna(0).astype("int64")
            .floordiv(2**12)
            .mod(2)
            .eq(1)
            .astype("boolean")
            .mask(flags.isna())
        )

    required = {
        "source_id",
        "transit_id",
        "g_transit_time",
        "g_transit_mag",
        "g_transit_flux",
        "g_transit_flux_error",
        "variability_flag_g_reject",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing photometry columns: {sorted(missing)}"
        )

    return data


def get_sources():
    if SOURCE_FILE.exists():
        return pd.read_csv(SOURCE_FILE)

    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    tables = []

    for class_name, class_table in CLASS_TABLES.items():
        query = f"""
        SELECT TOP {SAMPLE_SIZE}
            source.source_id,
            source.random_index,
            source.ra,
            source.dec,
            source.ecl_lon,
            source.ecl_lat,
            source.phot_g_mean_mag AS g_mean_magnitude,
            source.phot_bp_mean_mag AS bp_mean_magnitude,
            source.phot_rp_mean_mag AS rp_mean_magnitude,
            source.bp_rp AS bp_rp_color,
            {PERIOD_COLUMNS[class_name]}
        FROM gaiadr3.{class_table} AS variability
        JOIN gaiadr3.gaia_source AS source
            ON variability.source_id = source.source_id
        WHERE source.has_epoch_photometry = 'true'
            AND {PERIOD_FILTERS[class_name]}
        ORDER BY source.random_index
        """

        job = Gaia.launch_job_async(query)
        data = job.get_results().to_pandas()
        data["class_name"] = class_name

        if len(data) != SAMPLE_SIZE:
            raise RuntimeError(
                f"{class_name}: Expected {SAMPLE_SIZE}, received {len(data)}."
            )

        tables.append(data)
        print(f"{class_name}: {len(data)} stars")

    sources = pd.concat(tables, ignore_index=True)
    sources["source_id"] = sources["source_id"].astype("int64")

    if sources["source_id"].duplicated().any():
        raise ValueError("Duplicate source IDs.")

    for col in [
        "gaia_pf",
        "gaia_p1_o",
        "gaia_p2_o",
        "gaia_frequency",
    ]:
        if col not in sources:
            sources[col] = np.nan

        sources[col] = pd.to_numeric(
            sources[col], errors="coerce"
        )

    sources["gaia_period"] = np.nan
    sources["gaia_period_origin"] = pd.NA

    for col, origin in [
        ("gaia_pf", "pf"),
        ("gaia_p1_o", "p1_o"),
        ("gaia_p2_o", "p2_o"),
    ]:
        mask = (
            sources["gaia_period"].isna()
            & sources[col].gt(0)
        )

        sources.loc[mask, "gaia_period"] = sources.loc[mask, col]
        sources.loc[mask, "gaia_period_origin"] = origin

    mask = (
        sources["gaia_period"].isna()
        & sources["gaia_frequency"].gt(0)
    )

    sources.loc[mask, "gaia_period"] = (
        1 / sources.loc[mask, "gaia_frequency"]
    )
    sources.loc[mask, "gaia_period_origin"] = "frequency"

    assert sources["gaia_period"].gt(0).all()

    sources.to_csv(SOURCE_FILE, index=False)

    return sources


def get_photometry(sources):
    if PHOTOMETRY_FILE.exists():
        return normalize(pd.read_csv(
            PHOTOMETRY_FILE,
            dtype={"transit_id": "string"},
        ))

    BATCH_ROOT.mkdir(parents=True, exist_ok=True)

    source_ids = sources["source_id"].astype("int64").tolist()
    tables = []

    for start in range(0, len(source_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(source_ids))
        batch_file = BATCH_ROOT / f"photometry_{start + 1}_{end}.csv"

        if batch_file.exists():
            batch_data = pd.read_csv(
                batch_file,
                dtype={"transit_id": "string"},
            )
        else:
            downloads = Gaia.load_data(
                ids=source_ids[start:end],
                data_release="Gaia DR3",
                retrieval_type="EPOCH_PHOTOMETRY",
                data_structure="DATAMODEL_STANDARD",
                format="csv",
            )

            batch_tables = [
                normalize(table.to_pandas())
                for group in downloads.values()
                for table in group
            ]

            if not batch_tables:
                raise RuntimeError(
                    f"No photometry for batch {start + 1}-{end}."
                )

            batch_data = pd.concat(
                batch_tables, ignore_index=True
            )

            batch_data.to_csv(batch_file, index=False)
            print(f"Photometry: {start + 1}-{end}")

            time.sleep(1)

        tables.append(normalize(batch_data))

    photometry = pd.concat(tables, ignore_index=True)
    photometry["source_id"] = photometry["source_id"].astype("int64")

    missing = set(source_ids) - set(photometry["source_id"])

    if missing:
        raise RuntimeError(
            f"Missing photometry for {len(missing)} stars."
        )

    photometry.to_csv(PHOTOMETRY_FILE, index=False)

    return photometry


def label(sources, photometry):
    return photometry.merge(
        sources,
        on="source_id",
        how="left",
        validate="many_to_one",
    )


def clean(data):
    data = normalize(data)

    flag = (
        data["variability_flag_g_reject"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    time = pd.to_numeric(
        data["g_transit_time"], errors="coerce"
    )

    magnitude = pd.to_numeric(
        data["g_transit_mag"], errors="coerce"
    )

    flux = pd.to_numeric(
        data["g_transit_flux"], errors="coerce"
    )

    valid = (
        flag.isin(["false", "0"])
        & np.isfinite(time)
        & np.isfinite(magnitude)
        & np.isfinite(flux)
        & flux.gt(0)
    )

    cleaned = data.loc[valid].copy()
    cleaned["g_transit_time"] = time.loc[valid]
    cleaned["g_transit_mag"] = magnitude.loc[valid]

    cleaned = cleaned.sort_values(["source_id", "g_transit_time"])

    known = cleaned["transit_id"].notna()
    duplicates = cleaned.loc[known].duplicated(["source_id", "transit_id"])
    cleaned = cleaned.drop(index=cleaned.loc[known].index[duplicates])
    return cleaned.drop_duplicates(["source_id", "g_transit_time"]).copy()

def get_data():
    sources = get_sources()
    photometry = get_photometry(sources)

    counts = clean(photometry).groupby("source_id").size()

    sources["n_g_epochs"] = (
        sources["source_id"]
        .map(counts)
        .fillna(0)
        .astype("int64")
    )

    sources.to_csv(SOURCE_FILE, index=False)

    return sources, photometry

def main():
    sources, photometry = get_data()
    labeled = label(sources, photometry)
    cleaned = clean(labeled)

    LABELED_FILE.parent.mkdir(parents=True, exist_ok=True)

    labeled.to_csv(LABELED_FILE, index=False)
    cleaned.to_csv(CLEAN_FILE, index=False)

    print("Stars:", len(sources))
    print("Photometry:", len(photometry))
    print("Clean Observations:", len(cleaned))

if __name__ == "__main__":
    main()