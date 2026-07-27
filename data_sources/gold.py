from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import requests

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

GOLD_SOURCE: dict[str, Any] = {
    "name": "DataHub Gold Prices",
    "url": (
        "https://datahub.io/core/gold-prices/"
        "_r/-/data/monthly.csv"
    ),
    "date_column": "Date",
    "price_column": "Price",
    "raw_file_name": "gold_raw.csv",
    "processed_file_name": "gold_price.csv",
}


# --------------------------------------------------
# Downloading
# --------------------------------------------------

def download_gold_data(
    url: str = GOLD_SOURCE["url"],
) -> pd.DataFrame:
    """
    Download the live monthly gold-price CSV.

    Args:
        url:
            URL of the source CSV file.

    Returns:
        The downloaded source data as a DataFrame.

    Raises:
        requests.RequestException:
            If the network request fails.

        ValueError:
            If the response cannot be parsed as CSV or
            contains no observations.
    """

    create_project_folders()

    print(
        f"\nDownloading gold prices from "
        f"{GOLD_SOURCE['name']}..."
    )

    response = requests.get(
        url,
        timeout=30,
    )

    response.raise_for_status()

    if not response.text.strip():
        raise ValueError(
            "The gold-price source returned an empty response."
        )

    try:
        raw_data = pd.read_csv(
            StringIO(response.text)
        )

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
    ) as error:
        raise ValueError(
            "The downloaded gold-price data could not "
            "be parsed as CSV."
        ) from error

    if raw_data.empty:
        raise ValueError(
            "The downloaded gold-price dataset is empty."
        )

    return raw_data


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

    Daily, monthly, or irregular observations are converted
    into monthly average prices.

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
    file_name: str = GOLD_SOURCE["raw_file_name"],
) -> Path:
    """
    Save the downloaded source data to the raw gold folder.

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
    file_name: str = GOLD_SOURCE[
        "processed_file_name"
    ],
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
# Processing and updating
# --------------------------------------------------

def process_gold_data(
    raw_data: pd.DataFrame,
    source_date_column: str = "date",
    source_value_column: str = "price",
) -> pd.DataFrame:
    """
    Save, normalize, validate, summarize, and process
    raw gold-price data.

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


def update_gold_data() -> pd.DataFrame:
    """
    Download, normalize, validate, and save the live
    gold-price dataset.

    Returns:
        Validated monthly gold-price data.
    """

    raw_data = download_gold_data()

    return process_gold_data(
        raw_data=raw_data,
        source_date_column=GOLD_SOURCE[
            "date_column"
        ],
        source_value_column=GOLD_SOURCE[
            "price_column"
        ],
    )


# --------------------------------------------------
# Program entry point
# --------------------------------------------------

def main() -> None:
    """
    Run the complete gold-price update pipeline.
    """

    print("\n" + "=" * 72)
    print("UPDATING GOLD DATA".center(72))
    print("=" * 72)

    try:
        update_gold_data()

    except requests.Timeout:
        print(
            "\nThe gold-price request timed out. "
            "Check your internet connection and try again."
        )

    except requests.HTTPError as error:
        print(
            "\nThe gold-price source returned an HTTP error."
        )

        if error.response is not None:
            print(
                f"Status code: "
                f"{error.response.status_code}"
            )

    except requests.RequestException as error:
        print(
            f"\nThe gold-price download failed: {error}"
        )

    except (
        ValueError,
        OSError,
    ) as error:
        print(
            f"\nThe gold update process failed: {error}"
        )

    else:
        print(
            "\nGold data updated successfully."
        )

    print("=" * 72)


if __name__ == "__main__":
    main()