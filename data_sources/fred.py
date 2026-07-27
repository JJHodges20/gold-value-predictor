from pathlib import Path
from typing import Any

import pandas as pd
import requests

from data_sources.config import (
    FRED_API_KEY,
    PROCESSED_DATA_FOLDER,
    RAW_FRED_FOLDER,
    create_project_folders,
    validate_configuration,
)
from data_sources.validator import (
    dataset_summary,
    validate_dataframe,
)


# --------------------------------------------------
# FRED API configuration
# --------------------------------------------------

FRED_OBSERVATIONS_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)


FRED_SERIES: dict[str, dict[str, Any]] = {
    "cpi": {
        "series_id": "CPIAUCSL",
        "column_name": "CPI",
        "aggregation_method": "avg",
        "minimum": 0,
        "maximum": None,
    },
    "fed_funds_rate": {
        "series_id": "FEDFUNDS",
        "column_name": "Fed Funds Rate",
        "aggregation_method": "avg",
        "minimum": 0,
        "maximum": 30,
    },
    "treasury_10_year": {
        "series_id": "GS10",
        "column_name": "10-Year Treasury Yield",
        "aggregation_method": "avg",
        "minimum": 0,
        "maximum": 30,
    },
    "unemployment_rate": {
        "series_id": "UNRATE",
        "column_name": "Unemployment Rate",
        "aggregation_method": "avg",
        "minimum": 0,
        "maximum": 100,
    },
    "recession_indicator": {
        "series_id": "USREC",
        "column_name": "Recession",
        "aggregation_method": "avg",
        "minimum": 0,
        "maximum": 1,
    },
    "oil_price": {
        "series_id": "DCOILWTICO",
        "column_name": "WTI Oil Price",
        "aggregation_method": "avg",
        "minimum": None,
        "maximum": None,
    },
    "sp500": {
        "series_id": "SP500",
        "column_name": "S&P 500",
        "aggregation_method": "eop",
        "minimum": 0,
        "maximum": None,
    },
}


# --------------------------------------------------
# Download functions
# --------------------------------------------------

def download_fred_series(
    series_id: str,
    column_name: str,
    aggregation_method: str = "avg",
) -> pd.DataFrame:
    """
    Download one FRED series and return normalized
    monthly data.

    Args:
        series_id:
            FRED series identifier, such as CPIAUCSL.

        column_name:
            Friendly output column name, such as CPI.

        aggregation_method:
            Method used when FRED converts higher-frequency
            observations to monthly frequency.

            Common values:
            - avg
            - sum
            - eop

    Returns:
        A normalized DataFrame containing:
        - Date
        - the configured value column

    Raises:
        requests.RequestException:
            If the HTTP request fails.

        ValueError:
            If FRED returns no observations or malformed data.

        RuntimeError:
            If project configuration is invalid.
    """

    validate_configuration()
    create_project_folders()

    parameters = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "frequency": "m",
        "aggregation_method": aggregation_method,
        "sort_order": "asc",
    }

    print(
        f"\nDownloading {column_name} "
        f"from FRED ({series_id})..."
    )

    response = requests.get(
        FRED_OBSERVATIONS_URL,
        params=parameters,
        timeout=30,
    )

    response.raise_for_status()

    try:
        response_data = response.json()
    except requests.JSONDecodeError as error:
        raise ValueError(
            f"FRED returned an invalid JSON response "
            f"for {series_id}."
        ) from error

    observations = response_data.get(
        "observations",
        [],
    )

    if not observations:
        error_message = response_data.get(
            "error_message",
            f"FRED returned no observations for {series_id}.",
        )

        raise ValueError(error_message)

    raw_data = pd.DataFrame(observations)

    raw_output_path = save_raw_fred_data(
        data=raw_data,
        series_id=series_id,
    )

    print(
        f"Saved raw response to "
        f"{raw_output_path.name}."
    )

    processed_data = normalize_fred_data(
        data=raw_data,
        column_name=column_name,
    )

    return processed_data


# --------------------------------------------------
# Data cleaning
# --------------------------------------------------

def normalize_fred_data(
    data: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    """
    Convert raw FRED observations into a clean,
    monthly dataset.

    Invalid dates and nonnumeric values are removed.
    Dates are converted to the first day of each month.
    Duplicate months are resolved by keeping the final
    observation for that month.

    Args:
        data:
            Raw FRED observations.

        column_name:
            Friendly name for the processed numeric column.

    Returns:
        A cleaned monthly DataFrame.

    Raises:
        ValueError:
            If required columns are missing or no valid rows
            remain after cleaning.
    """

    required_columns = {
        "date",
        "value",
    }

    if not required_columns.issubset(data.columns):
        raise ValueError(
            "The FRED response did not contain the expected "
            "'date' and 'value' columns."
        )

    normalized_data = data[
        ["date", "value"]
    ].copy()

    normalized_data = normalized_data.rename(
        columns={
            "date": "Date",
            "value": column_name,
        }
    )

    normalized_data["Date"] = pd.to_datetime(
        normalized_data["Date"],
        errors="coerce",
    )

    normalized_data[column_name] = pd.to_numeric(
        normalized_data[column_name],
        errors="coerce",
    )

    normalized_data = normalized_data.dropna(
        subset=[
            "Date",
            column_name,
        ]
    )

    normalized_data["Date"] = (
        normalized_data["Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    normalized_data = (
        normalized_data
        .sort_values("Date")
        .drop_duplicates(
            subset=["Date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    if normalized_data.empty:
        raise ValueError(
            "No valid observations remained after cleaning."
        )

    return normalized_data


# --------------------------------------------------
# File saving
# --------------------------------------------------

def save_raw_fred_data(
    data: pd.DataFrame,
    series_id: str,
) -> Path:
    """
    Save the original FRED response as a raw CSV file.

    Args:
        data:
            Raw FRED observations.

        series_id:
            FRED series identifier.

    Returns:
        The path of the saved raw CSV file.
    """

    create_project_folders()

    output_path = (
        RAW_FRED_FOLDER
        / f"{series_id.lower()}_raw.csv"
    )

    data.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_processed_fred_data(
    data: pd.DataFrame,
    file_name: str,
) -> Path:
    """
    Save validated FRED data to the processed folder.

    The in-memory DataFrame keeps full datetime values.
    The saved CSV stores dates in YYYY-MM format.

    Args:
        data:
            Validated FRED data.

        file_name:
            Output file name without the .csv extension.

    Returns:
        The path of the saved processed CSV file.
    """

    create_project_folders()

    output_path = (
        PROCESSED_DATA_FOLDER
        / f"{file_name}.csv"
    )

    output_data = data.copy()

    output_data["Date"] = (
        output_data["Date"]
        .dt.strftime("%Y-%m")
    )

    output_data.to_csv(
        output_path,
        index=False,
    )

    return output_path


# --------------------------------------------------
# Update functions
# --------------------------------------------------

def update_single_fred_series(
    series_id: str,
    column_name: str,
    file_name: str,
    aggregation_method: str = "avg",
    minimum: float | None = None,
    maximum: float | None = None,
) -> pd.DataFrame:
    """
    Download, normalize, validate, summarize, and save
    one FRED series.

    Args:
        series_id:
            FRED series identifier.

        column_name:
            Friendly processed column name.

        file_name:
            Processed CSV file name without an extension.

        aggregation_method:
            FRED monthly aggregation method.

        minimum:
            Optional minimum allowed value.

        maximum:
            Optional maximum allowed value.

    Returns:
        The validated processed DataFrame.
    """

    data = download_fred_series(
        series_id=series_id,
        column_name=column_name,
        aggregation_method=aggregation_method,
    )

    validate_dataframe(
        data=data,
        value_column=column_name,
        minimum=minimum,
        maximum=maximum,
    )

    summary = dataset_summary(
        data=data,
        value_column=column_name,
    )

    output_path = save_processed_fred_data(
        data=data,
        file_name=file_name,
    )

    first_date = summary[
        "first_date"
    ].strftime("%Y-%m")

    last_date = summary[
        "last_date"
    ].strftime("%Y-%m")

    print(
        f"Validation passed for {column_name}."
    )

    print(
        f"Saved {summary['rows']:,} monthly observations "
        f"to {output_path.name}."
    )

    print(
        f"Date range: {first_date} through {last_date}."
    )

    print(
        f"Value range: "
        f"{summary['minimum_value']:,.4f} through "
        f"{summary['maximum_value']:,.4f}."
    )

    return data


def update_all_fred_series() -> dict[str, pd.DataFrame]:
    """
    Download and save every configured FRED series.

    One failed series does not stop the remaining series
    from being processed.

    Returns:
        A dictionary of successfully updated DataFrames,
        keyed by configured file name.
    """

    updated_series: dict[str, pd.DataFrame] = {}
    failed_series: dict[str, str] = {}

    print("\n" + "=" * 72)
    print("UPDATING FRED DATA".center(72))
    print("=" * 72)

    for file_name, settings in FRED_SERIES.items():
        try:
            data = update_single_fred_series(
                series_id=settings["series_id"],
                column_name=settings["column_name"],
                file_name=file_name,
                aggregation_method=settings[
                    "aggregation_method"
                ],
                minimum=settings["minimum"],
                maximum=settings["maximum"],
            )

            updated_series[file_name] = data

        except (
            requests.RequestException,
            RuntimeError,
            ValueError,
            OSError,
        ) as error:
            failed_series[file_name] = str(error)

            print(
                f"\nCould not update {file_name}: {error}"
            )

    print("\n" + "-" * 72)
    print("FRED UPDATE SUMMARY".center(72))
    print("-" * 72)

    print(
        f"Successful updates: {len(updated_series)}"
    )

    print(
        f"Failed updates: {len(failed_series)}"
    )

    if updated_series:
        print("\nSuccessful datasets:")

        for file_name, data in updated_series.items():
            print(
                f"- {file_name}: {len(data):,} rows"
            )

    if failed_series:
        print("\nFailed datasets:")

        for file_name, error_message in failed_series.items():
            print(
                f"- {file_name}: {error_message}"
            )

    else:
        print(
            "\nAll FRED datasets updated successfully."
        )

    print("=" * 72)

    return updated_series


# --------------------------------------------------
# Program entry point
# --------------------------------------------------

def main() -> None:
    """
    Download, validate, and save all configured
    FRED datasets.
    """

    try:
        update_all_fred_series()

    except requests.Timeout:
        print(
            "\nA FRED request timed out. "
            "Check your internet connection and try again."
        )

    except requests.HTTPError as error:
        print(
            "\nFRED returned an HTTP error."
        )

        if error.response is not None:
            print(
                f"Status code: "
                f"{error.response.status_code}"
            )

            try:
                error_data = error.response.json()

                print(
                    "Details: "
                    f"{error_data.get('error_message', error_data)}"
                )

            except ValueError:
                print(
                    f"Details: {error.response.text}"
                )

    except requests.RequestException as error:
        print(
            f"\nThe FRED request failed: {error}"
        )

    except (
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        print(
            f"\nThe FRED update process failed: {error}"
        )


if __name__ == "__main__":
    main()