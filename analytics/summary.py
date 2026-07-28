from pathlib import Path
from typing import Any

import pandas as pd

from analytics.loader import load_master_data
from data_sources.config import PROCESSED_DATA_FOLDER
from data_sources.merger import DATE_COLUMN, MASTER_FILE_NAME


def calculate_column_summary(
    data: pd.DataFrame,
    column: str,
) -> dict[str, Any]:
    """
    Calculate descriptive and coverage statistics for one column.

    Args:
        data:
            Master dataset containing a Date column.

        column:
            Column to summarize.

    Returns:
        A dictionary containing availability, coverage, and
        descriptive statistics.

    Raises:
        ValueError:
            If the requested column does not exist, is the Date
            column, or contains no usable observations.
    """

    if column not in data.columns:
        raise ValueError(
            f"Column was not found in the dataset: {column}"
        )

    if column == DATE_COLUMN:
        raise ValueError(
            "The Date column cannot be summarized as an "
            "analytical value column."
        )

    available_mask = data[column].notna()
    available_values = data.loc[available_mask, column]

    total_rows = int(len(data))
    available_rows = int(available_mask.sum())
    missing_rows = total_rows - available_rows

    completeness_percent = (
        available_rows / total_rows * 100
        if total_rows > 0
        else 0.0
    )

    summary: dict[str, Any] = {
        "column": column,
        "data_type": str(data[column].dtype),
        "total_rows": total_rows,
        "available_rows": available_rows,
        "missing_rows": missing_rows,
        "completeness_percent": round(
            completeness_percent,
            2,
        ),
        "coverage_start": None,
        "coverage_end": None,
        "minimum": None,
        "maximum": None,
        "mean": None,
        "median": None,
        "standard_deviation": None,
    }

    if available_values.empty:
        return summary

    summary["coverage_start"] = (
        data.loc[available_mask, DATE_COLUMN]
        .min()
        .strftime("%Y-%m")
    )

    summary["coverage_end"] = (
        data.loc[available_mask, DATE_COLUMN]
        .max()
        .strftime("%Y-%m")
    )

    if pd.api.types.is_numeric_dtype(data[column]):
        standard_deviation = available_values.std()

        summary.update(
            {
                "minimum": float(
                    available_values.min()
                ),
                "maximum": float(
                    available_values.max()
                ),
                "mean": float(
                    available_values.mean()
                ),
                "median": float(
                    available_values.median()
                ),
                "standard_deviation": (
                    float(standard_deviation)
                    if pd.notna(standard_deviation)
                    else None
                ),
            }
        )

    return summary


def build_dataset_summary(
    data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a complete summary of the master dataset.

    Args:
        data:
            Validated master dataset.

    Returns:
        A dictionary containing dataset-level and
        column-level statistics.

    Raises:
        ValueError:
            If the dataset is empty or lacks a Date column.
    """

    if data.empty:
        raise ValueError(
            "Cannot summarize an empty dataset."
        )

    if DATE_COLUMN not in data.columns:
        raise ValueError(
            f"The dataset is missing the "
            f"{DATE_COLUMN} column."
        )

    analytical_columns = [
        column
        for column in data.columns
        if column != DATE_COLUMN
    ]

    numeric_column_names = [
        column
        for column in analytical_columns
        if pd.api.types.is_numeric_dtype(
            data[column]
        )
    ]

    non_numeric_column_names = [
        column
        for column in analytical_columns
        if column not in numeric_column_names
    ]

    column_summaries = {
        column: calculate_column_summary(
            data=data,
            column=column,
        )
        for column in analytical_columns
    }

    return {
        "row_count": int(len(data)),
        "column_count": int(len(data.columns)),
        "analytical_column_count": int(
            len(analytical_columns)
        ),
        "numeric_column_count": int(
            len(numeric_column_names)
        ),
        "non_numeric_column_count": int(
            len(non_numeric_column_names)
        ),
        "date_range": {
            "start": (
                data[DATE_COLUMN]
                .min()
                .strftime("%Y-%m")
            ),
            "end": (
                data[DATE_COLUMN]
                .max()
                .strftime("%Y-%m")
            ),
        },
        "available_columns": analytical_columns,
        "numeric_columns": numeric_column_names,
        "non_numeric_columns": (
            non_numeric_column_names
        ),
        "columns": column_summaries,
    }


def generate_dataset_summary(
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> dict[str, Any]:
    """
    Load the master dataset and generate its summary.

    Args:
        processed_folder:
            Folder containing the master dataset.

        file_name:
            Name of the master dataset CSV file.

    Returns:
        Complete dataset summary.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    return build_dataset_summary(data)


def format_number(
    value: float | int | None,
    decimal_places: int = 2,
) -> str:
    """
    Format a numeric value for console output.

    Args:
        value:
            Number to format.

        decimal_places:
            Number of decimal places to display for
            floating-point values.

    Returns:
        A formatted string or N/A for missing values.
    """

    if value is None:
        return "N/A"

    if isinstance(value, int):
        return f"{value:,}"

    return f"{value:,.{decimal_places}f}"


def print_dataset_summary(
    summary: dict[str, Any],
) -> None:
    """
    Print a readable master dataset summary.

    Args:
        summary:
            Summary generated by build_dataset_summary()
            or generate_dataset_summary().
    """

    separator = "=" * 72
    section_separator = "-" * 72

    print(separator)
    print("GOLD VALUE PREDICTOR — DATASET SUMMARY")
    print(separator)

    print("\nDATASET OVERVIEW")
    print(section_separator)
    print(
        f"Rows:                 "
        f"{summary['row_count']:,}"
    )
    print(
        f"Total columns:        "
        f"{summary['column_count']:,}"
    )
    print(
        f"Analytical columns:   "
        f"{summary['analytical_column_count']:,}"
    )
    print(
        f"Numeric columns:      "
        f"{summary['numeric_column_count']:,}"
    )
    print(
        f"Non-numeric columns:  "
        f"{summary['non_numeric_column_count']:,}"
    )
    print(
        f"Date range:           "
        f"{summary['date_range']['start']} "
        f"through "
        f"{summary['date_range']['end']}"
    )

    print("\nAVAILABLE INDICATORS")
    print(section_separator)

    for column in summary["available_columns"]:
        print(f"- {column}")

    print("\nCOLUMN DETAILS")
    print(section_separator)

    for column in summary["available_columns"]:
        details = summary["columns"][column]

        print(f"\n{column}")
        print(f"  Data type:          {details['data_type']}")
        print(
            f"  Available rows:     "
            f"{details['available_rows']:,}"
        )
        print(
            f"  Missing rows:       "
            f"{details['missing_rows']:,}"
        )
        print(
            f"  Completeness:       "
            f"{details['completeness_percent']:.2f}%"
        )
        print(
            f"  Coverage:           "
            f"{details['coverage_start'] or 'N/A'} "
            f"through "
            f"{details['coverage_end'] or 'N/A'}"
        )

        if column in summary["numeric_columns"]:
            print(
                f"  Minimum:            "
                f"{format_number(details['minimum'])}"
            )
            print(
                f"  Maximum:            "
                f"{format_number(details['maximum'])}"
            )
            print(
                f"  Mean:               "
                f"{format_number(details['mean'])}"
            )
            print(
                f"  Median:             "
                f"{format_number(details['median'])}"
            )
            print(
                f"  Standard deviation: "
                f"{format_number(details['standard_deviation'])}"
            )

    print(f"\n{separator}")


def main() -> None:
    """
    Generate and print the master dataset summary.
    """

    try:
        summary = generate_dataset_summary()
        print_dataset_summary(summary)

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as error:
        print(
            f"Unable to generate dataset summary: "
            f"{error}"
        )


if __name__ == "__main__":
    main()