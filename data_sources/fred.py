from pathlib import Path

import pandas as pd
import requests

from data_sources.config import (
    FRED_API_KEY,
    RAW_FRED_FOLDER,
    PROCESSED_DATA_FOLDER,
    create_project_folders,
    validate_configuration,
)


FRED_OBSERVATIONS_URL = (
    "https://api.stlouisfed.org/fred/series/observations"
)


def download_fred_series(
    series_id: str,
    column_name: str,
    aggregation_method: str = "avg",
) -> pd.DataFrame:
    """
    Download one FRED series and return normalized monthly data.

    Args:
        series_id:
            FRED series identifier, such as CPIAUCSL.

        column_name:
            Friendly output column name, such as CPI.

        aggregation_method:
            Method FRED should use when converting higher-frequency
            data to monthly frequency.

            Supported values include:
            - avg
            - sum
            - eop

    Returns:
        A DataFrame with:
        - Date
        - the requested value column
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
        f"Downloading {column_name} "
        f"from FRED ({series_id})..."
    )

    response = requests.get(
        FRED_OBSERVATIONS_URL,
        params=parameters,
        timeout=30,
    )

    response.raise_for_status()

    response_data = response.json()

    observations = response_data.get(
        "observations",
        [],
    )

    if not observations:
        raise ValueError(
            f"FRED returned no observations for {series_id}."
        )

    raw_data = pd.DataFrame(observations)

    save_raw_fred_data(
        data=raw_data,
        series_id=series_id,
    )

    processed_data = normalize_fred_data(
        data=raw_data,
        column_name=column_name,
    )

    return processed_data


def normalize_fred_data(
    data: pd.DataFrame,
    column_name: str,
) -> pd.DataFrame:
    """
    Convert raw FRED observations into a clean monthly dataset.
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


def save_raw_fred_data(
    data: pd.DataFrame,
    series_id: str,
) -> Path:
    """
    Save the unprocessed FRED response as a CSV file.
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
    Save normalized FRED data to the processed-data folder.
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


def update_single_fred_series(
    series_id: str,
    column_name: str,
    file_name: str,
    aggregation_method: str = "avg",
) -> pd.DataFrame:
    """
    Download, normalize, and save one FRED series.
    """

    data = download_fred_series(
        series_id=series_id,
        column_name=column_name,
        aggregation_method=aggregation_method,
    )

    output_path = save_processed_fred_data(
        data=data,
        file_name=file_name,
    )

    print(
        f"Saved {len(data):,} monthly observations "
        f"to {output_path}."
    )

    return data


def main() -> None:
    """
    Test the FRED downloader using CPI data.
    """

    try:
        cpi_data = update_single_fred_series(
            series_id="CPIAUCSL",
            column_name="CPI",
            file_name="cpi",
            aggregation_method="avg",
        )

        print("\nFirst five rows:")
        print(cpi_data.head())

        print("\nLast five rows:")
        print(cpi_data.tail())

    except requests.Timeout:
        print(
            "\nThe FRED request timed out. "
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
            f"\nFRED data update failed: {error}"
        )


if __name__ == "__main__":
    main()