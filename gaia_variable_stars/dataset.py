import time

import pandas
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


def get_sources():
    RAW_ROOT.mkdir(parents=True, exist_ok=True)

    if SOURCE_FILE.exists():
        return pandas.read_csv(SOURCE_FILE)

    source_tables = []

    for class_name, class_table in CLASS_TABLES.items():
        query = f"""
        SELECT TOP {SAMPLE_SIZE}
            source.source_id,
            source.random_index,
            source.phot_g_mean_mag AS g_mean_magnitude,
            source.phot_bp_mean_mag AS bp_mean_magnitude,
            source.phot_rp_mean_mag AS rp_mean_magnitude,
            source.bp_rp AS bp_rp_color
        FROM gaiadr3.{class_table} AS variability
        JOIN gaiadr3.gaia_source AS source
            ON variability.source_id = source.source_id
        WHERE source.has_epoch_photometry = 'true'
        ORDER BY source.random_index
        """

        job = Gaia.launch_job_async(query)
        class_data = job.get_results().to_pandas()
        class_data["class_name"] = class_name

        source_tables.append(class_data)

    source_data = pandas.concat(source_tables, ignore_index=True)
    source_data["source_id"] = source_data["source_id"].astype("int64")
    source_data.to_csv(SOURCE_FILE, index=False)

    return source_data


def get_photometry(source_data):
    BATCH_ROOT.mkdir(parents=True, exist_ok=True)

    source_numbers = source_data["source_id"].astype("int64").tolist()
    batch_files = []

    for start in range(0, len(source_numbers), BATCH_SIZE):
        batch = source_numbers[start:start + BATCH_SIZE]
        end = min(start + BATCH_SIZE, len(source_numbers))
        batch_file = BATCH_ROOT / f"photometry_{start + 1}_{end}.csv"

        batch_files.append(batch_file)

        if batch_file.exists():
            continue

        downloads = Gaia.load_data(
            ids=batch,
            data_release="Gaia DR3",
            retrieval_type="EPOCH_PHOTOMETRY",
            data_structure="INDIVIDUAL",
            format="csv",
        )

        light_tables = []

        for table_group in downloads.values():
            for light_table in table_group:
                light_tables.append(light_table.to_pandas())

        if light_tables:
            batch_data = pandas.concat(light_tables, ignore_index=True)
            batch_data.to_csv(batch_file, index=False)

        time.sleep(1)

    light_tables = [
        pandas.read_csv(batch_file)
        for batch_file in batch_files
        if batch_file.exists()
    ]

    photometry_data = pandas.concat(light_tables, ignore_index=True)
    photometry_data["source_id"] = photometry_data["source_id"].astype("int64")
    photometry_data.to_csv(PHOTOMETRY_FILE, index=False)

    return photometry_data


def get_data():
    source_data = get_sources()
    photometry_data = get_photometry(source_data)

    return source_data, photometry_data


def label(sources, photometry):
    """Attach class label and source columns to every epoch."""
    return photometry.merge(
        sources,
        on="source_id",
        how="left",
        validate="many_to_one",
    )


def clean(data):
    """Drop epochs with missing time or magnitude, rejected epochs, and duplicates."""
    bad = (
        data["variability_flag_g_reject"]
        .astype("string")
        .str.lower()
        .isin(["true", "1"])
    )
    keep = data["g_transit_time"].notna() & data["g_transit_mag"].notna() & ~bad

    return data[keep].drop_duplicates(["source_id", "transit_id"]).copy()


def main():
    sources, photometry = get_data()
    data = label(sources, photometry)

    LABELED_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(LABELED_FILE, index=False)
    clean(data).to_csv(CLEAN_FILE, index=False)

if __name__ == "__main__":
    # get_data()
    main()

def main():
    sources, photometry = get_data()
    data = label(sources, photometry)

    LABELED_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(LABELED_FILE, index=False)
    clean(data).to_csv(CLEAN_FILE, index=False)


if __name__ == "__main__":
    # get_data()
    main()