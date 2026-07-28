from pathlib import Path
from typing import Any

import pandas as pd

from data_sources.config import PROCESSED_DATA_FOLDER
from data_sources.merger import (
    DATE_COLUMN,
    MASTER_FILE_NAME,
)


def load_master_data(
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
    copy: bool = True,
) -> pd.DataFrame:
    """
    Load and validate the project's master dataset.

    The Date column is converted to monthly timestamps,
    duplicate months are rejected, and the dataset is
    returned in chronological order.

    Args:
        processed_folder:
            Folder containing the master dataset.

        file_name:
            Name of the master dataset CSV file.

        copy:
            Whether to return a copy of the loaded
            DataFrame.

    Returns:
        A validated and chronologically sorted DataFrame.

    Raises:
        FileNotFoundError:
            If the master dataset does not exist.

        ValueError:
            If the file is empty, lacks a Date column,
            contains invalid dates, or contains duplicate
            monthly observations.

        OSError:
            If the file cannot be read.
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

    except pd.errors.ParserError as error:
        raise ValueError(
            "The master dataset could not be parsed."
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

    duplicate_mask = data[DATE_COLUMN].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicate_dates = (
            data.loc[
                duplicate_mask,
                DATE_COLUMN,
            ]
            .dt.strftime("%Y-%m")
            .unique()
            .tolist()
        )

        duplicate_text = ", ".join(
            duplicate_dates
        )

        raise ValueError(
            "The master dataset contains duplicate "
            f"monthly dates: {duplicate_text}"
        )

    data = (
        data
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    if copy:
        return data.copy(deep=True)

    return data


def parse_month(
    value: str | pd.Timestamp,
    parameter_name: str,
) -> pd.Timestamp:
    """
    Parse a date-range boundary as a monthly timestamp.

    Args:
        value:
            Date value to parse.

        parameter_name:
            Name used in validation error messages.

    Returns:
        The first day of the parsed month.

    Raises:
        ValueError:
            If the supplied value is not a valid date.
    """

    try:
        parsed_value = pd.to_datetime(
            value,
            errors="raise",
        )

    except (
        ValueError,
        TypeError,
        pd.errors.ParserError,
    ) as error:
        raise ValueError(
            f"Invalid {parameter_name} date: {value}"
        ) from error

    return (
        parsed_value
        .to_period("M")
        .to_timestamp()
    )


def load_date_range(
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> pd.DataFrame:
    """
    Load observations within an inclusive monthly range.

    Either boundary may be omitted. When start is None,
    the dataset's earliest date is used. When end is
    None, the latest date is used.

    Args:
        start:
            Optional beginning month.

        end:
            Optional ending month.

        processed_folder:
            Folder containing the master dataset.

        file_name:
            Name of the master dataset CSV file.

    Returns:
        A filtered copy of the master dataset.

    Raises:
        ValueError:
            If a boundary is invalid, start occurs after
            end, or the range contains no observations.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    start_date = (
        parse_month(
            value=start,
            parameter_name="start",
        )
        if start is not None
        else data[DATE_COLUMN].min()
    )

    end_date = (
        parse_month(
            value=end,
            parameter_name="end",
        )
        if end is not None
        else data[DATE_COLUMN].max()
    )

    if start_date > end_date:
        raise ValueError(
            "The start date cannot occur after the "
            "end date."
        )

    date_mask = (
        data[DATE_COLUMN].between(
            start_date,
            end_date,
            inclusive="both",
        )
    )

    filtered_data = (
        data.loc[date_mask]
        .reset_index(drop=True)
        .copy(deep=True)
    )

    if filtered_data.empty:
        raise ValueError(
            "No observations were found within the "
            "requested date range."
        )

    return filtered_data


def available_columns(
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> list[str]:
    """
    Return every analytical column in the master dataset.

    The Date column is excluded.

    Args:
        processed_folder:
            Folder containing the master dataset.

        file_name:
            Name of the master dataset CSV file.

    Returns:
        A list of available analytical columns.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    return [
        column
        for column in data.columns
        if column != DATE_COLUMN
    ]


def numeric_columns(
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> list[str]:
    """
    Return the numeric analytical columns.

    The Date column is excluded even if its underlying
    representation is numeric.

    Args:
        processed_folder:
            Folder containing the master dataset.

        file_name:
            Name of the master dataset CSV file.

    Returns:
        A list of numeric column names.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    return [
        column
        for column in data.columns
        if (
            column != DATE_COLUMN
            and pd.api.types.is_numeric_dtype(
                data[column]
            )
        )
    ]


def dataset_date_range(
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> dict[str, str]:
    """
    Return the first and last months in the master data.

    Args:
        processed_folder:
            Folder containing the master dataset.

        file_name:
            Name of the master dataset CSV file.

    Returns:
        A dictionary containing start and end months.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    return {
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
    }


def column_summary(
    column: str,
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> dict[str, Any]:
    """
    Calculate descriptive statistics for one numeric
    column in the master dataset.

    Missing values are excluded from numeric statistics
    but are included in the missing-value count.

    Args:
        column:
            Numeric column to summarize.

        processed_folder:
            Folder containing the master dataset.

        file_name:
            Name of the master dataset CSV file.

    Returns:
        Descriptive statistics and coverage information.

    Raises:
        ValueError:
            If the requested column does not exist or has
            no usable observations.

        TypeError:
            If the requested column is not numeric.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    if column not in data.columns:
        raise ValueError(
            f"Column was not found in the master dataset: "
            f"{column}"
        )

    if column == DATE_COLUMN:
        raise ValueError(
            "The Date column cannot be summarized as an "
            "analytical value column."
        )

    if not pd.api.types.is_numeric_dtype(
        data[column]
    ):
        raise TypeError(
            f"Column is not numeric: {column}"
        )

    values = data[column].dropna()

    if values.empty:
        raise ValueError(
            f"Column contains no usable observations: "
            f"{column}"
        )

    available_mask = data[column].notna()

    coverage_start = (
        data.loc[
            available_mask,
            DATE_COLUMN,
        ]
        .min()
        .strftime("%Y-%m")
    )

    coverage_end = (
        data.loc[
            available_mask,
            DATE_COLUMN,
        ]
        .max()
        .strftime("%Y-%m")
    )

    count = int(values.count())
    missing = int(data[column].isna().sum())
    total_rows = int(len(data))

    completeness = (
        count / total_rows * 100
        if total_rows > 0
        else 0.0
    )

    standard_deviation = values.std()

    return {
        "column": column,
        "count": count,
        "missing": missing,
        "completeness_percent": round(
            completeness,
            2,
        ),
        "coverage_start": coverage_start,
        "coverage_end": coverage_end,
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
        "standard_deviation": (
            float(standard_deviation)
            if pd.notna(standard_deviation)
            else None
        ),
    }