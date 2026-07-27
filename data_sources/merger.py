import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from data_sources.config import (
    PROCESSED_DATA_FOLDER,
    create_project_folders,
)


# --------------------------------------------------
# Output configuration
# --------------------------------------------------

DATE_COLUMN = "Date"

MASTER_FILE_NAME = "master_dataset.csv"
METADATA_FILE_NAME = "master_dataset_metadata.json"


# --------------------------------------------------
# Dataset configuration
# --------------------------------------------------
#
# output_column is the standardized name that will
# appear in master_dataset.csv.
#
# The loader automatically detects the source value
# column inside each CSV, so it does not need to match
# the output column exactly.
# --------------------------------------------------

DATASETS: list[dict[str, str]] = [
    {
        "name": "Gold",
        "file_name": "gold_price.csv",
        "output_column": "Gold Price",
    },
    {
        "name": "CPI",
        "file_name": "cpi.csv",
        "output_column": "CPI",
    },
    {
        "name": "Fed Funds Rate",
        "file_name": "fed_funds_rate.csv",
        "output_column": "Fed Funds Rate",
    },
    {
        "name": "10-Year Treasury",
        "file_name": "treasury_10_year.csv",
        "output_column": "10-Year Treasury",
    },
    {
        "name": "Unemployment",
        "file_name": "unemployment_rate.csv",
        "output_column": "Unemployment Rate",
    },
    {
        "name": "Recession Indicator",
        "file_name": "recession_indicator.csv",
        "output_column": "Recession Indicator",
    },
    {
        "name": "WTI Oil",
        "file_name": "oil_price.csv",
        "output_column": "WTI Oil Price",
    },
    {
        "name": "S&P 500",
        "file_name": "sp500.csv",
        "output_column": "S&P 500",
    },
]


# --------------------------------------------------
# Loading
# --------------------------------------------------

def load_processed_dataset(
    dataset_config: dict[str, str],
    processed_folder: Path = PROCESSED_DATA_FOLDER,
) -> pd.DataFrame:
    """
    Load and validate one processed dataset.

    The source value column is detected automatically.
    Each processed CSV must contain a Date column and
    exactly one additional data column.

    Args:
        dataset_config:
            Configuration containing:
            - name
            - file_name
            - output_column

        processed_folder:
            Folder containing processed CSV files.

    Returns:
        A DataFrame containing Date and the standardized
        output column.

    Raises:
        FileNotFoundError:
            If the configured file does not exist.

        ValueError:
            If the dataset is empty, has no Date column,
            contains an unexpected number of value columns,
            contains invalid dates, or has duplicate months.
    """

    dataset_name = dataset_config["name"]
    file_name = dataset_config["file_name"]
    output_column = dataset_config["output_column"]

    file_path = processed_folder / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"{dataset_name} dataset was not found: "
            f"{file_path}"
        )

    try:
        data = pd.read_csv(file_path)

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            f"{dataset_name} dataset contains no readable data."
        ) from error

    if data.empty:
        raise ValueError(
            f"{dataset_name} dataset is empty."
        )

    if DATE_COLUMN not in data.columns:
        available_columns = ", ".join(
            str(column)
            for column in data.columns
        )

        raise ValueError(
            f"{dataset_name} dataset is missing the "
            f"{DATE_COLUMN} column. Available columns: "
            f"{available_columns}."
        )

    value_columns = [
        column
        for column in data.columns
        if column != DATE_COLUMN
    ]

    if len(value_columns) != 1:
        available_columns = ", ".join(
            str(column)
            for column in data.columns
        )

        raise ValueError(
            f"{dataset_name} dataset must contain exactly "
            f"one value column in addition to Date. "
            f"Available columns: {available_columns}."
        )

    source_value_column = value_columns[0]

    dataset = data[
        [
            DATE_COLUMN,
            source_value_column,
        ]
    ].copy()

    dataset = dataset.rename(
        columns={
            source_value_column: output_column,
        }
    )

    parsed_dates = pd.to_datetime(
        dataset[DATE_COLUMN],
        errors="coerce",
    )

    invalid_date_count = int(
        parsed_dates.isna().sum()
    )

    if invalid_date_count > 0:
        raise ValueError(
            f"{dataset_name} dataset contains "
            f"{invalid_date_count} invalid date value(s)."
        )

    dataset[DATE_COLUMN] = (
        parsed_dates
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    duplicate_count = int(
        dataset[DATE_COLUMN]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{dataset_name} dataset contains "
            f"{duplicate_count} duplicate monthly date(s)."
        )

    dataset[output_column] = pd.to_numeric(
        dataset[output_column],
        errors="coerce",
    )

    dataset = (
        dataset
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    return dataset


# --------------------------------------------------
# Merging
# --------------------------------------------------

def merge_processed_datasets(
    datasets: list[dict[str, str]] = DATASETS,
    processed_folder: Path = PROCESSED_DATA_FOLDER,
) -> pd.DataFrame:
    """
    Load and merge every configured processed dataset.

    An outer join preserves the full historical range
    available from every source.

    Args:
        datasets:
            Configured datasets to merge.

        processed_folder:
            Folder containing processed CSV files.

    Returns:
        A merged monthly DataFrame.

    Raises:
        ValueError:
            If no datasets are configured, duplicate output
            columns exist, or the result is empty.
    """

    if not datasets:
        raise ValueError(
            "No datasets are configured for merging."
        )

    output_columns = [
        dataset["output_column"]
        for dataset in datasets
    ]

    duplicate_output_columns = {
        column
        for column in output_columns
        if output_columns.count(column) > 1
    }

    if duplicate_output_columns:
        duplicate_text = ", ".join(
            sorted(duplicate_output_columns)
        )

        raise ValueError(
            f"Duplicate output columns are configured: "
            f"{duplicate_text}."
        )

    master_data: pd.DataFrame | None = None

    for dataset_config in datasets:
        dataset_name = dataset_config["name"]

        print(
            f"Loading {dataset_name}..."
        )

        dataset = load_processed_dataset(
            dataset_config=dataset_config,
            processed_folder=processed_folder,
        )

        if master_data is None:
            master_data = dataset

        else:
            master_data = master_data.merge(
                dataset,
                on=DATE_COLUMN,
                how="outer",
                validate="one_to_one",
            )

    if master_data is None or master_data.empty:
        raise ValueError(
            "The master dataset is empty."
        )

    master_data = (
        master_data
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    if master_data[DATE_COLUMN].duplicated().any():
        raise ValueError(
            "The master dataset contains duplicate dates."
        )

    return master_data


# --------------------------------------------------
# Metadata
# --------------------------------------------------

def create_dataset_metadata(
    master_data: pd.DataFrame,
    datasets: list[dict[str, str]] = DATASETS,
) -> dict[str, Any]:
    """
    Create JSON-compatible metadata for the master
    dataset.
    """

    if master_data.empty:
        raise ValueError(
            "Cannot create metadata for an empty dataset."
        )

    first_date = master_data[DATE_COLUMN].min()
    last_date = master_data[DATE_COLUMN].max()

    missing_values = {
        column: int(
            master_data[column].isna().sum()
        )
        for column in master_data.columns
        if column != DATE_COLUMN
    }

    source_files = {
        dataset["name"]: dataset["file_name"]
        for dataset in datasets
    }

    return {
        "generated_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "rows": int(len(master_data)),
        "columns": int(len(master_data.columns)),
        "datasets": [
            dataset["name"]
            for dataset in datasets
        ],
        "source_files": source_files,
        "date_range": {
            "start": first_date.strftime("%Y-%m"),
            "end": last_date.strftime("%Y-%m"),
        },
        "missing_values": missing_values,
    }


# --------------------------------------------------
# Saving
# --------------------------------------------------

def save_master_dataset(
    master_data: pd.DataFrame,
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> Path:
    """
    Save the master dataset using YYYY-MM dates.
    """

    create_project_folders()

    processed_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = processed_folder / file_name

    output_data = master_data.copy()

    output_data[DATE_COLUMN] = (
        output_data[DATE_COLUMN]
        .dt.strftime("%Y-%m")
    )

    output_data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_dataset_metadata(
    metadata: dict[str, Any],
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = METADATA_FILE_NAME,
) -> Path:
    """
    Save master-dataset metadata as formatted JSON.
    """

    create_project_folders()

    processed_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = processed_folder / file_name

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as metadata_file:
        json.dump(
            metadata,
            metadata_file,
            indent=2,
        )

    return output_path


# --------------------------------------------------
# Complete pipeline
# --------------------------------------------------

def build_master_dataset(
    datasets: list[dict[str, str]] = DATASETS,
    processed_folder: Path = PROCESSED_DATA_FOLDER,
) -> pd.DataFrame:
    """
    Merge the processed datasets and save the master
    CSV and metadata JSON.
    """

    print("\nBuilding master dataset...\n")

    master_data = merge_processed_datasets(
        datasets=datasets,
        processed_folder=processed_folder,
    )

    metadata = create_dataset_metadata(
        master_data=master_data,
        datasets=datasets,
    )

    master_path = save_master_dataset(
        master_data=master_data,
        processed_folder=processed_folder,
    )

    metadata_path = save_dataset_metadata(
        metadata=metadata,
        processed_folder=processed_folder,
    )

    print(
        f"\nSaved {metadata['rows']:,} monthly rows "
        f"to {master_path.name}."
    )

    print(
        f"Date range: "
        f"{metadata['date_range']['start']} through "
        f"{metadata['date_range']['end']}."
    )

    print(
        f"Saved dataset metadata to "
        f"{metadata_path.name}."
    )

    return master_data


def main() -> None:
    """
    Run the master-dataset build process.
    """

    print("\n" + "=" * 72)
    print("MASTER DATASET BUILDER".center(72))
    print("=" * 72)

    try:
        build_master_dataset()

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        pd.errors.ParserError,
        pd.errors.MergeError,
    ) as error:
        print(
            f"\nMaster dataset build failed: {error}"
        )

    else:
        print(
            "\nMaster dataset built successfully."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()