import pandas as pd
import pytest

from data_sources.fred import normalize_fred_data


def test_normalize_fred_data() -> None:
    """Raw FRED observations should become clean monthly data."""

    raw_data = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
            ],
            "value": [
                "100.5",
                "101.2",
                "102.0",
            ],
        }
    )

    result = normalize_fred_data(
        data=raw_data,
        column_name="Test Value",
    )

    assert list(result.columns) == [
        "Date",
        "Test Value",
    ]

    assert len(result) == 3

    assert pd.api.types.is_datetime64_any_dtype(
        result["Date"]
    )

    assert pd.api.types.is_numeric_dtype(
        result["Test Value"]
    )

    assert result.iloc[0]["Test Value"] == 100.5


def test_normalize_fred_data_removes_invalid_rows() -> None:
    """Invalid dates and values should be removed."""

    raw_data = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "not-a-date",
                "2024-03-01",
            ],
            "value": [
                "100.5",
                "101.2",
                ".",
            ],
        }
    )

    result = normalize_fred_data(
        data=raw_data,
        column_name="Test Value",
    )

    assert len(result) == 1
    assert result.iloc[0]["Test Value"] == 100.5


def test_normalize_fred_data_rejects_missing_columns() -> None:
    """A malformed FRED response should raise an error."""

    invalid_data = pd.DataFrame(
        {
            "wrong_column": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="expected",
    ):
        normalize_fred_data(
            data=invalid_data,
            column_name="Test Value",
        )


def test_normalize_fred_data_rejects_empty_result() -> None:
    """The function should reject data with no usable observations."""

    invalid_data = pd.DataFrame(
        {
            "date": [
                "not-a-date",
            ],
            "value": [
                ".",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="No valid observations",
    ):
        normalize_fred_data(
            data=invalid_data,
            column_name="Test Value",
        )