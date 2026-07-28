import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from data_sources.config import PROCESSED_DATA_FOLDER
from data_sources.merger import (
    DATE_COLUMN,
    MASTER_FILE_NAME,
    METADATA_FILE_NAME,
)


def load_master_dataset(
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> pd.DataFrame:
    """
    Load the saved master dataset and parse its Date column.

    Args:
        processed_folder:
            Folder containing the master dataset.

        file_name:
            Name of the master CSV file.

    Returns:
        The loaded master dataset.

    Raises:
        FileNotFoundError:
            If the master dataset does not exist.

        ValueError:
            If the dataset is empty, missing its Date
            column, or contains invalid dates.
    """

    file_path = processed_folder / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Master dataset was not found: {file_path}"
        )

    try:
        data = pd.read_csv(file_path)

    except pd.errors.EmptyDataError as error:
        raise ValueError(
            "The master dataset contains no readable data."
        ) from error

    if data.empty:
        raise ValueError(
            "The master dataset is empty."
        )

    if DATE_COLUMN not in data.columns:
        raise ValueError(
            f"The master dataset is missing the "
            f"{DATE_COLUMN} column."
        )

    parsed_dates = pd.to_datetime(
        data[DATE_COLUMN],
        errors="coerce",
    )

    invalid_date_count = int(
        parsed_dates.isna().sum()
    )

    if invalid_date_count > 0:
        raise ValueError(
            f"The master dataset contains "
            f"{invalid_date_count} invalid date value(s)."
        )

    data[DATE_COLUMN] = (
        parsed_dates
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    data = (
        data
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    return data


def load_master_metadata(
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = METADATA_FILE_NAME,
) -> dict[str, Any]:
    """
    Load the master-dataset metadata JSON file.

    Args:
        processed_folder:
            Folder containing the metadata file.

        file_name:
            Name of the metadata JSON file.

    Returns:
        Parsed metadata.

    Raises:
        FileNotFoundError:
            If the metadata file does not exist.

        ValueError:
            If the file contains invalid JSON or does not
            contain a JSON object.
    """

    file_path = processed_folder / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Master metadata was not found: {file_path}"
        )

    try:
        with file_path.open(
            mode="r",
            encoding="utf-8",
        ) as metadata_file:
            metadata = json.load(metadata_file)

    except json.JSONDecodeError as error:
        raise ValueError(
            "The master metadata file contains invalid JSON."
        ) from error

    if not isinstance(metadata, dict):
        raise ValueError(
            "The master metadata must contain a JSON object."
        )

    return metadata


def calculate_column_health(
    data: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Calculate coverage and completeness information for
    one master-dataset column.

    Args:
        data:
            Loaded master dataset.

        column:
            Column to inspect.

    Returns:
        Health information for the selected column.

    Raises:
        ValueError:
            If the column does not exist.
    """

    if column not in data.columns:
        raise ValueError(
            f"Column was not found in the master dataset: "
            f"{column}"
        )

    total_rows = int(len(data))

    available_mask = data[column].notna()
    available_rows = int(available_mask.sum())
    missing_rows = int(data[column].isna().sum())

    completeness = (
        available_rows / total_rows * 100
        if total_rows > 0
        else 0.0
    )

    available_data = data.loc[
        available_mask,
        [
            DATE_COLUMN,
            column,
        ],
    ]

    if available_data.empty:
        coverage_start = None
        coverage_end = None

    else:
        coverage_start = (
            available_data[DATE_COLUMN]
            .min()
            .strftime("%Y-%m")
        )

        coverage_end = (
            available_data[DATE_COLUMN]
            .max()
            .strftime("%Y-%m")
        )

    return {
        "column": column,
        "available_rows": available_rows,
        "missing_rows": missing_rows,
        "completeness_percent": round(
            completeness,
            2,
        ),
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
    }


def build_dataset_health_report(
    data: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    Build a complete health report for the master dataset.

    Args:
        data:
            Loaded master dataset.

        metadata:
            Loaded metadata JSON.

    Returns:
        A JSON-compatible health report.
    """

    if data.empty:
        raise ValueError(
            "Cannot build a health report for an empty dataset."
        )

    value_columns = [
        column
        for column in data.columns
        if column != DATE_COLUMN
    ]

    column_health = {
        column: calculate_column_health(
            data=data,
            column=column,
        )
        for column in value_columns
    }

    overall_start = (
        data[DATE_COLUMN]
        .min()
        .strftime("%Y-%m")
    )

    overall_end = (
        data[DATE_COLUMN]
        .max()
        .strftime("%Y-%m")
    )

    return {
        "generated_at": metadata.get(
            "generated_at"
        ),
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "overall_date_range": {
            "start": overall_start,
            "end": overall_end,
        },
        "datasets": metadata.get(
            "datasets",
            [],
        ),
        "source_files": metadata.get(
            "source_files",
            {},
        ),
        "column_health": column_health,
    }


def format_generated_at(
    generated_at: str | None,
) -> str:
    """
    Convert an ISO timestamp into a readable UTC value.
    """

    if not generated_at:
        return "Unknown"

    try:
        parsed_value = datetime.fromisoformat(
            generated_at.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:
        return generated_at

    return parsed_value.strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def print_dataset_health_report(
    report: dict[str, Any],
) -> None:
    """
    Print the dataset health report in a readable format.
    """

    print("\n" + "=" * 72)
    print("DATASET HEALTH REPORT".center(72))
    print("=" * 72)

    generated_at = format_generated_at(
        report.get("generated_at")
    )

    date_range = report[
        "overall_date_range"
    ]

    print(
        f"\nGenerated: {generated_at}"
    )

    print(
        f"Rows: {report['rows']:,}"
    )

    print(
        f"Columns: {report['columns']:,}"
    )

    print(
        f"Overall range: "
        f"{date_range['start']} through "
        f"{date_range['end']}"
    )

    column_health = report[
        "column_health"
    ]

    for column, health in column_health.items():
        print("\n" + "-" * 72)
        print(column)
        print("-" * 72)

        coverage_start = health[
            "coverage_start"
        ]

        coverage_end = health[
            "coverage_end"
        ]

        if (
            coverage_start is None
            or coverage_end is None
        ):
            coverage_text = "No available observations"

        else:
            coverage_text = (
                f"{coverage_start} through "
                f"{coverage_end}"
            )

        print(
            f"Coverage: {coverage_text}"
        )

        print(
            f"Available rows: "
            f"{health['available_rows']:,}"
        )

        print(
            f"Missing rows: "
            f"{health['missing_rows']:,}"
        )

        print(
            f"Completeness: "
            f"{health['completeness_percent']:.2f}%"
        )

    print("\n" + "=" * 72)


def generate_dataset_health_report(
    processed_folder: Path = PROCESSED_DATA_FOLDER,
) -> dict[str, Any]:
    """
    Load the saved master files, build the health report,
    and print it.

    Returns:
        The completed health report.
    """

    data = load_master_dataset(
        processed_folder=processed_folder,
    )

    metadata = load_master_metadata(
        processed_folder=processed_folder,
    )

    report = build_dataset_health_report(
        data=data,
        metadata=metadata,
    )

    print_dataset_health_report(
        report
    )

    return report


def main() -> None:
    """
    Run the dataset health-report process.
    """

    try:
        generate_dataset_health_report()

    except (
        FileNotFoundError,
        ValueError,
        OSError,
        pd.errors.ParserError,
    ) as error:
        print("\n" + "=" * 72)
        print("DATASET HEALTH REPORT".center(72))
        print("=" * 72)

        print(
            f"\nDataset health report failed: {error}"
        )

        print("\n" + "=" * 72)


if __name__ == "__main__":
    main()