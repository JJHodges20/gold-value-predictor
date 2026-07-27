from pathlib import Path

import pandas as pd

from data_sources.config import (
    PROCESSED_DATA_FOLDER,
    RAW_GOLD_FOLDER,
    create_project_folders,
)
from data_sources.validator import (
    dataset_summary,
    validate_dataframe,
)


# --------------------------------------------------
# Gold data configuration
# --------------------------------------------------

GOLD_DATE_COLUMN = "Date"
GOLD_VALUE_COLUMN = "Gold Price"


# --------------------------------------------------
# Data normalization
# --------------------------------------------------

def normalize_gold_data(
    data: pd.DataFrame,
    source_date_column: str = "date",
    source_value_column: str = "price",
) -> pd.DataFrame:
    """
    Normalize raw gold-price data into one monthly
    observation per row.

    Daily or irregular observations are converted into
    monthly average prices.

    Args:
        data:
            Raw gold-price observations.

        source_date_column:
            Name of the source date column.

        source_value_column:
            Name of the source price column.

    Returns:
        A DataFrame containing:
        - Date
        - Gold Price

    Raises:
        ValueError:
            If required columns are missing or no valid
            observations remain after cleaning.
    """

    required_columns = {
        source_date_column,
        source_value_column,
    }

    if not required_columns.issubset(data.columns):
        missing_columns = sorted(
            required_columns - set(data.columns)
        )

        missing_text = ", ".join(missing_columns)

        raise ValueError(
            f"Gold data is missing required columns: "
            f"{missing_text}."
        )

    normalized_data = data[
        [
            source_date_column,
            source_value_column,
        ]
    ].copy()

    normalized_data = normalized_data.rename(
        columns={
            source_date_column: GOLD_DATE_COLUMN,
            source_value_column: GOLD_VALUE_COLUMN,
        }
    )

    normalized_data[GOLD_DATE_COLUMN] = pd.to_datetime(
        normalized_data[GOLD_DATE_COLUMN],
        errors="coerce",
    )

    normalized_data[GOLD_VALUE_COLUMN] = pd.to_numeric(
        normalized_data[GOLD_VALUE_COLUMN],
        errors="coerce",
    )

    normalized_data = normalized_data.dropna(
        subset=[
            GOLD_DATE_COLUMN,
            GOLD_VALUE_COLUMN,
        ]
    )

    normalized_data = normalized_data[
        normalized_data[GOLD_VALUE_COLUMN] > 0
    ]

    if normalized_data.empty:
        raise ValueError(
            "No valid gold-price observations remained "
            "after cleaning."
        )

    normalized_data[GOLD_DATE_COLUMN] = (
        normalized_data[GOLD_DATE_COLUMN]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly_data = (
        normalized_data
        .groupby(
            GOLD_DATE_COLUMN,
            as_index=False,
        )[GOLD_VALUE_COLUMN]
        .mean()
        .sort_values(GOLD_DATE_COLUMN)
        .reset_index(drop=True)
    )

    monthly_data[GOLD_VALUE_COLUMN] = (
        monthly_data[GOLD_VALUE_COLUMN]
        .round(2)
    )

    if monthly_data.empty:
        raise ValueError(
            "No monthly gold-price observations were produced."
        )

    return monthly_data


# --------------------------------------------------
# File saving
# --------------------------------------------------

def save_raw_gold_data(
    data: pd.DataFrame,
    file_name: str = "gold_raw.csv",
) -> Path:
    """
    Save the untouched source data to the raw gold folder.

    Args:
        data:
            Raw gold-price data.

        file_name:
            Name of the raw CSV file.

    Returns:
        Path to the saved file.
    """

    create_project_folders()

    output_path = RAW_GOLD_FOLDER / file_name

    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_processed_gold_data(
    data: pd.DataFrame,
    file_name: str = "gold_price.csv",
) -> Path:
    """
    Save validated monthly gold-price data.

    The in-memory DataFrame retains datetime values.
    The CSV stores dates in YYYY-MM format.

    Args:
        data:
            Validated monthly gold data.

        file_name:
            Name of the processed CSV file.

    Returns:
        Path to the saved file.
    """

    create_project_folders()

    output_path = PROCESSED_DATA_FOLDER / file_name

    output_data = data.copy()

    output_data[GOLD_DATE_COLUMN] = (
        output_data[GOLD_DATE_COLUMN]
        .dt.strftime("%Y-%m")
    )

    output_data.to_csv(
        output_path,
        index=False,
    )

    return output_path


# --------------------------------------------------
# Processing
# --------------------------------------------------

def process_gold_data(
    raw_data: pd.DataFrame,
    source_date_column: str = "date",
    source_value_column: str = "price",
) -> pd.DataFrame:
    """
    Normalize, validate, summarize, and save raw
    gold-price data.

    Args:
        raw_data:
            Raw gold-price observations.

        source_date_column:
            Source column containing dates.

        source_value_column:
            Source column containing prices.

    Returns:
        Validated monthly gold-price data.
    """

    raw_output_path = save_raw_gold_data(
        data=raw_data,
    )

    print(
        f"Saved raw gold data to "
        f"{raw_output_path.name}."
    )

    normalized_data = normalize_gold_data(
        data=raw_data,
        source_date_column=source_date_column,
        source_value_column=source_value_column,
    )

    validate_dataframe(
        data=normalized_data,
        value_column=GOLD_VALUE_COLUMN,
        minimum=0,
    )

    summary = dataset_summary(
        data=normalized_data,
        value_column=GOLD_VALUE_COLUMN,
    )

    processed_output_path = save_processed_gold_data(
        data=normalized_data,
    )

    first_date = summary[
        "first_date"
    ].strftime("%Y-%m")

    last_date = summary[
        "last_date"
    ].strftime("%Y-%m")

    print("Gold data validation passed.")

    print(
        f"Saved {summary['rows']:,} monthly observations "
        f"to {processed_output_path.name}."
    )

    print(
        f"Date range: {first_date} through {last_date}."
    )

    print(
        f"Gold-price range: "
        f"${summary['minimum_value']:,.2f} through "
        f"${summary['maximum_value']:,.2f}."
    )

    return normalized_data