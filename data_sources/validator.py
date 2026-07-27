from typing import Any

import pandas as pd


def validate_required_columns(
    data: pd.DataFrame,
    required_columns: list[str],
) -> None:
    """
    Confirm that all required columns exist.

    Raises:
        ValueError: If one or more required columns are missing.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        raise ValueError(
            f"Missing required columns: {missing_text}."
        )


def validate_not_empty(
    data: pd.DataFrame,
) -> None:
    """
    Confirm that the dataset contains at least one row.

    Raises:
        ValueError: If the dataset is empty.
    """

    if data.empty:
        raise ValueError(
            "The dataset is empty."
        )


def validate_date_column(
    data: pd.DataFrame,
    date_column: str = "Date",
) -> None:
    """
    Confirm that the date column contains valid datetime values.

    Raises:
        ValueError: If the date column contains invalid values
        or is not stored as a datetime type.
    """

    validate_required_columns(
        data=data,
        required_columns=[date_column],
    )

    if not pd.api.types.is_datetime64_any_dtype(
        data[date_column]
    ):
        raise ValueError(
            f"The '{date_column}' column must use a "
            "datetime data type."
        )

    if data[date_column].isna().any():
        raise ValueError(
            f"The '{date_column}' column contains "
            "missing or invalid dates."
        )


def validate_duplicate_dates(
    data: pd.DataFrame,
    date_column: str = "Date",
) -> None:
    """
    Confirm that each date appears only once.

    Raises:
        ValueError: If duplicate dates are found.
    """

    validate_required_columns(
        data=data,
        required_columns=[date_column],
    )

    duplicate_mask = data[date_column].duplicated(
        keep=False
    )

    if duplicate_mask.any():
        duplicate_dates = (
            data.loc[duplicate_mask, date_column]
            .dt.strftime("%Y-%m")
            .drop_duplicates()
            .tolist()
        )

        duplicate_text = ", ".join(duplicate_dates)

        raise ValueError(
            f"Duplicate dates found: {duplicate_text}."
        )


def validate_sorted_dates(
    data: pd.DataFrame,
    date_column: str = "Date",
) -> None:
    """
    Confirm that dates are sorted in ascending order.

    Raises:
        ValueError: If dates are out of order.
    """

    validate_required_columns(
        data=data,
        required_columns=[date_column],
    )

    if not data[date_column].is_monotonic_increasing:
        raise ValueError(
            f"The '{date_column}' column is not sorted "
            "in ascending order."
        )


def validate_no_missing_values(
    data: pd.DataFrame,
    columns: list[str],
) -> None:
    """
    Confirm that selected columns contain no missing values.

    Raises:
        ValueError: If missing values are found.
    """

    validate_required_columns(
        data=data,
        required_columns=columns,
    )

    columns_with_missing_values = [
        column
        for column in columns
        if data[column].isna().any()
    ]

    if columns_with_missing_values:
        missing_text = ", ".join(
            columns_with_missing_values
        )

        raise ValueError(
            f"Missing values found in columns: "
            f"{missing_text}."
        )


def validate_numeric_column(
    data: pd.DataFrame,
    numeric_column: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    """
    Confirm that a column is numeric and falls within
    optional value limits.

    Args:
        data:
            Dataset being validated.

        numeric_column:
            Name of the numeric column.

        minimum:
            Optional minimum allowed value.

        maximum:
            Optional maximum allowed value.

    Raises:
        ValueError: If the column is nonnumeric or contains
        values outside the configured range.
    """

    validate_required_columns(
        data=data,
        required_columns=[numeric_column],
    )

    if not pd.api.types.is_numeric_dtype(
        data[numeric_column]
    ):
        raise ValueError(
            f"The '{numeric_column}' column must be numeric."
        )

    if data[numeric_column].isna().any():
        raise ValueError(
            f"The '{numeric_column}' column contains "
            "missing values."
        )

    if (
        minimum is not None
        and (data[numeric_column] < minimum).any()
    ):
        raise ValueError(
            f"The '{numeric_column}' column contains values "
            f"below the minimum of {minimum}."
        )

    if (
        maximum is not None
        and (data[numeric_column] > maximum).any()
    ):
        raise ValueError(
            f"The '{numeric_column}' column contains values "
            f"above the maximum of {maximum}."
        )


def validate_dataframe(
    data: pd.DataFrame,
    value_column: str,
    date_column: str = "Date",
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    """
    Run the standard project validation checks.

    Raises:
        ValueError: If any validation rule fails.
    """

    validate_not_empty(data)

    validate_required_columns(
        data=data,
        required_columns=[
            date_column,
            value_column,
        ],
    )

    validate_date_column(
        data=data,
        date_column=date_column,
    )

    validate_duplicate_dates(
        data=data,
        date_column=date_column,
    )

    validate_sorted_dates(
        data=data,
        date_column=date_column,
    )

    validate_no_missing_values(
        data=data,
        columns=[
            date_column,
            value_column,
        ],
    )

    validate_numeric_column(
        data=data,
        numeric_column=value_column,
        minimum=minimum,
        maximum=maximum,
    )


def dataset_summary(
    data: pd.DataFrame,
    value_column: str,
    date_column: str = "Date",
) -> dict[str, Any]:
    """
    Return a summary of a validated dataset.
    """

    validate_dataframe(
        data=data,
        value_column=value_column,
        date_column=date_column,
    )

    return {
        "rows": len(data),
        "first_date": data[date_column].min(),
        "last_date": data[date_column].max(),
        "minimum_value": data[value_column].min(),
        "maximum_value": data[value_column].max(),
    }