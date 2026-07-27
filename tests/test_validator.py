import pandas as pd
import pytest

from data_sources.validator import (
    dataset_summary,
    validate_dataframe,
    validate_duplicate_dates,
    validate_no_missing_values,
    validate_not_empty,
    validate_numeric_column,
    validate_required_columns,
    validate_sorted_dates,
)


@pytest.fixture
def valid_data() -> pd.DataFrame:
    """
    Return a valid monthly dataset for testing.
    """

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                ]
            ),
            "CPI": [
                308.4,
                309.7,
                310.3,
            ],
        }
    )


def test_validate_not_empty_rejects_empty_data() -> None:
    data = pd.DataFrame()

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        validate_not_empty(data)


def test_validate_required_columns_rejects_missing_column(
    valid_data: pd.DataFrame,
) -> None:
    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_required_columns(
            data=valid_data,
            required_columns=[
                "Date",
                "Gold Price",
            ],
        )


def test_validate_duplicate_dates_rejects_duplicates() -> None:
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-01",
                ]
            ),
            "CPI": [
                308.4,
                309.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate dates",
    ):
        validate_duplicate_dates(data)


def test_validate_sorted_dates_rejects_unsorted_data() -> None:
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-02-01",
                    "2024-01-01",
                ]
            ),
            "CPI": [
                309.7,
                308.4,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="not sorted",
    ):
        validate_sorted_dates(data)


def test_validate_no_missing_values_rejects_nan() -> None:
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
            "CPI": [
                308.4,
                None,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing values",
    ):
        validate_no_missing_values(
            data=data,
            columns=[
                "Date",
                "CPI",
            ],
        )


def test_validate_numeric_column_rejects_text_values() -> None:
    data = pd.DataFrame(
        {
            "CPI": [
                "308.4",
                "309.7",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        validate_numeric_column(
            data=data,
            numeric_column="CPI",
        )


def test_validate_numeric_column_rejects_low_values() -> None:
    data = pd.DataFrame(
        {
            "CPI": [
                308.4,
                -1.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="below the minimum",
    ):
        validate_numeric_column(
            data=data,
            numeric_column="CPI",
            minimum=0,
        )


def test_validate_dataframe_accepts_valid_data(
    valid_data: pd.DataFrame,
) -> None:
    validate_dataframe(
        data=valid_data,
        value_column="CPI",
        minimum=0,
    )


def test_dataset_summary_returns_expected_values(
    valid_data: pd.DataFrame,
) -> None:
    summary = dataset_summary(
        data=valid_data,
        value_column="CPI",
    )

    assert summary["rows"] == 3
    assert summary["first_date"] == pd.Timestamp(
        "2024-01-01"
    )
    assert summary["last_date"] == pd.Timestamp(
        "2024-03-01"
    )
    assert summary["minimum_value"] == 308.4
    assert summary["maximum_value"] == 310.3