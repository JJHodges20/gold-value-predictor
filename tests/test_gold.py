import pandas as pd
import pytest

from data_sources.gold import (
    GOLD_DATE_COLUMN,
    GOLD_VALUE_COLUMN,
    normalize_gold_data,
)


def test_normalize_gold_data() -> None:
    """
    Valid daily observations should become monthly
    average gold prices.
    """

    raw_data = pd.DataFrame(
        {
            "date": [
                "2024-01-02",
                "2024-01-15",
                "2024-02-01",
                "2024-02-20",
            ],
            "price": [
                "2000.00",
                "2020.00",
                "2040.00",
                "2060.00",
            ],
        }
    )

    result = normalize_gold_data(raw_data)

    assert list(result.columns) == [
        GOLD_DATE_COLUMN,
        GOLD_VALUE_COLUMN,
    ]

    assert len(result) == 2

    assert result.iloc[0][GOLD_DATE_COLUMN] == pd.Timestamp(
        "2024-01-01"
    )

    assert result.iloc[0][GOLD_VALUE_COLUMN] == 2010.00

    assert result.iloc[1][GOLD_VALUE_COLUMN] == 2050.00


def test_normalize_gold_data_removes_invalid_rows() -> None:
    """
    Invalid dates, missing prices, and nonnumeric prices
    should be removed.
    """

    raw_data = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "invalid-date",
                "2024-02-01",
                "2024-03-01",
            ],
            "price": [
                "2000.00",
                "2010.00",
                "not-a-price",
                None,
            ],
        }
    )

    result = normalize_gold_data(raw_data)

    assert len(result) == 1
    assert result.iloc[0][GOLD_VALUE_COLUMN] == 2000.00


def test_normalize_gold_data_rejects_missing_columns() -> None:
    """
    Missing source columns should raise an error.
    """

    raw_data = pd.DataFrame(
        {
            "wrong_date": [
                "2024-01-01",
            ],
            "wrong_price": [
                2000.00,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing required columns",
    ):
        normalize_gold_data(raw_data)


def test_normalize_gold_data_rejects_empty_result() -> None:
    """
    Completely invalid source data should raise an error.
    """

    raw_data = pd.DataFrame(
        {
            "date": [
                "invalid-date",
            ],
            "price": [
                "not-a-price",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="No valid gold-price observations",
    ):
        normalize_gold_data(raw_data)


def test_normalize_gold_data_rejects_nonpositive_prices() -> None:
    """
    Zero and negative gold prices should not remain.
    """

    raw_data = pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-02-01",
            ],
            "price": [
                0,
                -100,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="No valid gold-price observations",
    ):
        normalize_gold_data(raw_data)


def test_normalize_gold_data_supports_custom_columns() -> None:
    """
    The normalizer should support different source
    column names.
    """

    raw_data = pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-02-01",
            ],
            "USD": [
                2000.00,
                2050.00,
            ],
        }
    )

    result = normalize_gold_data(
        data=raw_data,
        source_date_column="Date",
        source_value_column="USD",
    )

    assert len(result) == 2
    assert result.iloc[1][GOLD_VALUE_COLUMN] == 2050.00