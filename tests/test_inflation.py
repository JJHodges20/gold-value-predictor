from pathlib import Path

import pandas as pd
import pytest

from analytics.inflation import (
    DEFAULT_CPI_COLUMN,
    MONTHS_PER_YEAR,
    align_price_and_cpi,
    calculate_annualized_inflation,
    calculate_inflation_adjustment_factor,
    calculate_monthly_inflation,
    prepare_cpi_series,
    resolve_base_cpi,
    validate_base_date,
)


def sample_inflation_data() -> pd.DataFrame:
    """
    Return a predictable monthly price-and-CPI dataset.

    Gold prices and CPI values are intentionally simple so
    expected inflation and adjustment calculations can be
    verified directly.
    """

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-06-01",
                ]
            ),
            "Gold Price": [
                100.0,
                110.0,
                121.0,
                115.0,
                126.5,
                139.15,
            ],
            "CPI": [
                200.0,
                202.0,
                204.0,
                206.0,
                208.0,
                210.0,
            ],
        }
    )


def unsorted_inflation_data() -> pd.DataFrame:
    """
    Return valid inflation data with dates out of order.
    """

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                [
                    "2024-03-01",
                    "2024-01-01",
                    "2024-02-01",
                ]
            ),
            "Gold Price": [
                121.0,
                100.0,
                110.0,
            ],
            "CPI": [
                204.0,
                200.0,
                202.0,
            ],
        }
    )


def custom_column_data() -> pd.DataFrame:
    """
    Return data using custom price and inflation column names.
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
            "Silver Price": [
                20.0,
                22.0,
                24.0,
            ],
            "Consumer Index": [
                100.0,
                101.0,
                102.0,
            ],
        }
    )


def write_master_csv(
    folder: Path,
    data: pd.DataFrame,
    file_name: str = "master_dataset.csv",
) -> Path:
    """
    Write a temporary master dataset for generator tests used
    later in this test module.
    """

    file_path = folder / file_name

    data.to_csv(
        file_path,
        index=False,
    )

    return file_path


# ------------------------------------------------------------------
# validate_base_date
# ------------------------------------------------------------------


def test_validate_base_date_accepts_none() -> None:
    """
    None should remain None so the latest CPI can be used.
    """

    assert validate_base_date(None) is None


def test_validate_base_date_accepts_date_string() -> None:
    """
    A valid date string should be converted to a Timestamp.
    """

    result = validate_base_date("2024-03-15")

    assert isinstance(result, pd.Timestamp)
    assert result == pd.Timestamp("2024-03-01")


def test_validate_base_date_normalizes_to_month_start() -> None:
    """
    Base dates should be normalized to the first day of their
    calendar month.
    """

    result = validate_base_date("2024-06-29")

    assert result == pd.Timestamp("2024-06-01")


def test_validate_base_date_accepts_timestamp() -> None:
    """
    A pandas Timestamp should be accepted and normalized.
    """

    result = validate_base_date(
        pd.Timestamp("2024-05-20")
    )

    assert result == pd.Timestamp("2024-05-01")


@pytest.mark.parametrize(
    "invalid_base_date",
    [
        202401,
        2024.01,
        [],
        {},
        True,
    ],
)
def test_validate_base_date_rejects_invalid_types(
    invalid_base_date: object,
) -> None:
    """
    Unsupported base-date types should raise TypeError.
    """

    with pytest.raises(
        TypeError,
        match="base date must be",
    ):
        validate_base_date(invalid_base_date)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_base_date",
    [
        "not-a-date",
        "2024-99-99",
        "February 40, 2024",
    ],
)
def test_validate_base_date_rejects_unparseable_strings(
    invalid_base_date: str,
) -> None:
    """
    Invalid date strings should raise a clear ValueError.
    """

    with pytest.raises(
        ValueError,
        match="Unable to parse base date",
    ):
        validate_base_date(invalid_base_date)


# ------------------------------------------------------------------
# prepare_cpi_series
# ------------------------------------------------------------------


def test_prepare_cpi_series_returns_series() -> None:
    """
    Valid CPI data should be returned as a pandas Series.
    """

    result = prepare_cpi_series(
        data=sample_inflation_data()
    )

    assert isinstance(result, pd.Series)


def test_prepare_cpi_series_uses_datetime_index() -> None:
    """
    Prepared CPI data should be indexed by date.
    """

    result = prepare_cpi_series(
        data=sample_inflation_data()
    )

    assert isinstance(
        result.index,
        pd.DatetimeIndex,
    )


def test_prepare_cpi_series_preserves_values() -> None:
    """
    Prepared CPI values should match the source dataset.
    """

    result = prepare_cpi_series(
        data=sample_inflation_data()
    )

    assert result.tolist() == pytest.approx(
        [
            200.0,
            202.0,
            204.0,
            206.0,
            208.0,
            210.0,
        ]
    )


def test_prepare_cpi_series_has_expected_name() -> None:
    """
    The resulting Series should retain the CPI column name.
    """

    result = prepare_cpi_series(
        data=sample_inflation_data()
    )

    assert result.name == DEFAULT_CPI_COLUMN


def test_prepare_cpi_series_sorts_dates() -> None:
    """
    CPI observations should be placed in chronological order.
    """

    result = prepare_cpi_series(
        data=unsorted_inflation_data()
    )

    expected_index = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
        ]
    )

    pd.testing.assert_index_equal(
        result.index,
        expected_index,
        check_names=False,
    )

    assert result.tolist() == pytest.approx(
        [
            200.0,
            202.0,
            204.0,
        ]
    )


def test_prepare_cpi_series_supports_custom_column() -> None:
    """
    Alternate inflation-index column names should be supported.
    """

    result = prepare_cpi_series(
        data=custom_column_data(),
        column="Consumer Index",
    )

    assert result.name == "Consumer Index"

    assert result.tolist() == pytest.approx(
        [
            100.0,
            101.0,
            102.0,
        ]
    )


def test_prepare_cpi_series_rejects_numeric_strings() -> None:
    """
    CPI values stored as strings should be rejected because
    the source column must already have a numeric dtype.
    """

    data = pd.DataFrame(
        {
            "Date": [
                "2024-01-01",
                "2024-02-01",
                "2024-03-01",
            ],
            "CPI": [
                "200.0",
                "202.0",
                "204.0",
            ],
        }
    )

    with pytest.raises(
        TypeError,
        match="Column is not numeric: CPI",
    ):
        prepare_cpi_series(data=data)


def test_prepare_cpi_series_rejects_negative_values() -> None:
    """
    Negative CPI values should not be accepted.
    """

    data = sample_inflation_data()

    data.loc[2, "CPI"] = -204.0

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        prepare_cpi_series(data=data)


def test_prepare_cpi_series_rejects_non_dataframe() -> None:
    """
    The CPI preparation function requires a DataFrame.
    """

    with pytest.raises(TypeError):
        prepare_cpi_series(  # type: ignore[arg-type]
            data=[200.0, 202.0, 204.0]
        )


def test_prepare_cpi_series_rejects_empty_dataframe() -> None:
    """
    A DataFrame without CPI observations should fail.
    """

    data = pd.DataFrame(
        {
            "Date": pd.Series(dtype="datetime64[ns]"),
            "CPI": pd.Series(dtype="float64"),
        }
    )

    with pytest.raises(ValueError):
        prepare_cpi_series(data=data)


# ------------------------------------------------------------------
# align_price_and_cpi
# ------------------------------------------------------------------


def test_align_price_and_cpi_returns_dataframe() -> None:
    """
    Aligned price and CPI data should be returned as a
    DataFrame.
    """

    result = align_price_and_cpi(
        data=sample_inflation_data()
    )

    assert isinstance(result, pd.DataFrame)


def test_align_price_and_cpi_uses_datetime_index() -> None:
    """
    Aligned data should use dates as its index.
    """

    result = align_price_and_cpi(
        data=sample_inflation_data()
    )

    assert isinstance(
        result.index,
        pd.DatetimeIndex,
    )


def test_align_price_and_cpi_contains_expected_columns() -> None:
    """
    The aligned result should contain price and CPI columns.
    """

    result = align_price_and_cpi(
        data=sample_inflation_data()
    )

    assert result.columns.tolist() == [
        "Gold Price",
        "CPI",
    ]


def test_align_price_and_cpi_preserves_prices() -> None:
    """
    Aligned nominal prices should match the source data.
    """

    result = align_price_and_cpi(
        data=sample_inflation_data()
    )

    assert result["Gold Price"].tolist() == pytest.approx(
        [
            100.0,
            110.0,
            121.0,
            115.0,
            126.5,
            139.15,
        ]
    )


def test_align_price_and_cpi_preserves_cpi_values() -> None:
    """
    Aligned CPI observations should match the source data.
    """

    result = align_price_and_cpi(
        data=sample_inflation_data()
    )

    assert result["CPI"].tolist() == pytest.approx(
        [
            200.0,
            202.0,
            204.0,
            206.0,
            208.0,
            210.0,
        ]
    )


def test_align_price_and_cpi_sorts_dates() -> None:
    """
    Aligned observations should be chronological.
    """

    result = align_price_and_cpi(
        data=unsorted_inflation_data()
    )

    expected_index = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
        ]
    )

    pd.testing.assert_index_equal(
        result.index,
        expected_index,
        check_names=False,
    )


def test_align_price_and_cpi_supports_custom_columns() -> None:
    """
    Custom price and CPI column names should be supported.
    """

    result = align_price_and_cpi(
        data=custom_column_data(),
        price_column="Silver Price",
        cpi_column="Consumer Index",
    )

    assert result.columns.tolist() == [
        "Silver Price",
        "Consumer Index",
    ]

    assert result["Silver Price"].tolist() == pytest.approx(
        [
            20.0,
            22.0,
            24.0,
        ]
    )

    assert result["Consumer Index"].tolist() == pytest.approx(
        [
            100.0,
            101.0,
            102.0,
        ]
    )


def test_align_price_and_cpi_rejects_missing_price_column() -> None:
    """
    A missing price column should prevent alignment.
    """

    data = sample_inflation_data().drop(
        columns=["Gold Price"]
    )

    with pytest.raises(ValueError):
        align_price_and_cpi(data=data)


def test_align_price_and_cpi_rejects_missing_cpi_column() -> None:
    """
    A missing CPI column should prevent alignment.
    """

    data = sample_inflation_data().drop(
        columns=["CPI"]
    )

    with pytest.raises(ValueError):
        align_price_and_cpi(data=data)


def test_align_price_and_cpi_rejects_invalid_cpi() -> None:
    """
    Invalid CPI values should be caught during alignment.
    """

    data = sample_inflation_data()

    data.loc[0, "CPI"] = 0.0

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        align_price_and_cpi(data=data)

# ------------------------------------------------------------------
# resolve_base_cpi
# ------------------------------------------------------------------


def test_resolve_base_cpi_uses_latest_observation_by_default() -> None:
    """
    When no base date is supplied, the latest CPI observation
    should be selected.
    """

    cpi = prepare_cpi_series(
        data=sample_inflation_data()
    )

    resolved_date, resolved_cpi = resolve_base_cpi(
        cpi=cpi
    )

    assert resolved_date == pd.Timestamp("2024-06-01")
    assert resolved_cpi == pytest.approx(210.0)


def test_resolve_base_cpi_uses_requested_month() -> None:
    """
    A requested month should resolve to its corresponding CPI
    observation.
    """

    cpi = prepare_cpi_series(
        data=sample_inflation_data()
    )

    resolved_date, resolved_cpi = resolve_base_cpi(
        cpi=cpi,
        base_date="2024-03-01",
    )

    assert resolved_date == pd.Timestamp("2024-03-01")
    assert resolved_cpi == pytest.approx(204.0)


def test_resolve_base_cpi_normalizes_requested_date() -> None:
    """
    A date within an available month should resolve to that
    month's CPI observation.
    """

    cpi = prepare_cpi_series(
        data=sample_inflation_data()
    )

    resolved_date, resolved_cpi = resolve_base_cpi(
        cpi=cpi,
        base_date="2024-04-27",
    )

    assert resolved_date == pd.Timestamp("2024-04-01")
    assert resolved_cpi == pytest.approx(206.0)


def test_resolve_base_cpi_accepts_timestamp() -> None:
    """
    A pandas Timestamp should be accepted as the requested
    base period.
    """

    cpi = prepare_cpi_series(
        data=sample_inflation_data()
    )

    resolved_date, resolved_cpi = resolve_base_cpi(
        cpi=cpi,
        base_date=pd.Timestamp("2024-05-15"),
    )

    assert resolved_date == pd.Timestamp("2024-05-01")
    assert resolved_cpi == pytest.approx(208.0)


def test_resolve_base_cpi_rejects_unavailable_month() -> None:
    """
    A requested month outside the CPI Series should fail.
    """

    cpi = prepare_cpi_series(
        data=sample_inflation_data()
    )

    with pytest.raises(
        ValueError,
        match="requested base date is not available",
    ):
        resolve_base_cpi(
            cpi=cpi,
            base_date="2023-12-01",
        )


def test_resolve_base_cpi_rejects_invalid_date_type() -> None:
    """
    Unsupported base-date values should be rejected through
    base-date validation.
    """

    cpi = prepare_cpi_series(
        data=sample_inflation_data()
    )

    with pytest.raises(TypeError):
        resolve_base_cpi(
            cpi=cpi,
            base_date=202406,  # type: ignore[arg-type]
        )


def test_resolve_base_cpi_rejects_invalid_date_string() -> None:
    """
    An unparseable requested base date should fail clearly.
    """

    cpi = prepare_cpi_series(
        data=sample_inflation_data()
    )

    with pytest.raises(
        ValueError,
        match="Unable to parse base date",
    ):
        resolve_base_cpi(
            cpi=cpi,
            base_date="not-a-date",
        )


# ------------------------------------------------------------------
# calculate_monthly_inflation
# ------------------------------------------------------------------


def test_calculate_monthly_inflation_returns_series() -> None:
    """
    Monthly inflation should be returned as a pandas Series.
    """

    result = calculate_monthly_inflation(
        data=sample_inflation_data()
    )

    assert isinstance(result, pd.Series)


def test_calculate_monthly_inflation_uses_datetime_index() -> None:
    """
    Monthly inflation results should remain indexed by date.
    """

    result = calculate_monthly_inflation(
        data=sample_inflation_data()
    )

    assert isinstance(
        result.index,
        pd.DatetimeIndex,
    )


def test_calculate_monthly_inflation_first_value_is_missing() -> None:
    """
    The first inflation rate should be missing because no
    previous CPI observation exists.
    """

    result = calculate_monthly_inflation(
        data=sample_inflation_data()
    )

    assert pd.isna(result.iloc[0])


def test_calculate_monthly_inflation_values() -> None:
    """
    Monthly inflation should equal percentage changes in CPI.
    """

    result = calculate_monthly_inflation(
        data=sample_inflation_data()
    )

    assert result.iloc[1] == pytest.approx(
        (202.0 / 200.0) - 1
    )

    assert result.iloc[2] == pytest.approx(
        (204.0 / 202.0) - 1
    )

    assert result.iloc[5] == pytest.approx(
        (210.0 / 208.0) - 1
    )


def test_calculate_monthly_inflation_name() -> None:
    """
    The inflation Series should have a descriptive name.
    """

    result = calculate_monthly_inflation(
        data=sample_inflation_data()
    )

    assert (
        result.name
        == "CPI Monthly Inflation Rate"
    )


def test_calculate_monthly_inflation_supports_custom_column() -> None:
    """
    Monthly inflation should support a custom CPI column.
    """

    result = calculate_monthly_inflation(
        data=custom_column_data(),
        cpi_column="Consumer Index",
    )

    assert result.iloc[1] == pytest.approx(
        (101.0 / 100.0) - 1
    )

    assert result.iloc[2] == pytest.approx(
        (102.0 / 101.0) - 1
    )

    assert (
        result.name
        == "Consumer Index Monthly Inflation Rate"
    )


def test_calculate_monthly_inflation_sorts_dates() -> None:
    """
    Monthly inflation should be calculated chronologically,
    even when the input rows are unsorted.
    """

    result = calculate_monthly_inflation(
        data=unsorted_inflation_data()
    )

    expected_index = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
        ]
    )

    pd.testing.assert_index_equal(
        result.index,
        expected_index,
        check_names=False,
    )

    assert result.iloc[1] == pytest.approx(
        (202.0 / 200.0) - 1
    )


def test_calculate_monthly_inflation_rejects_invalid_cpi() -> None:
    """
    Invalid CPI values should be rejected before calculating
    inflation.
    """

    data = sample_inflation_data()

    data.loc[2, "CPI"] = 0.0

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        calculate_monthly_inflation(data=data)


# ------------------------------------------------------------------
# calculate_annualized_inflation
# ------------------------------------------------------------------


def test_calculate_annualized_inflation_with_two_periods() -> None:
    """
    Period inflation should compare each CPI observation with
    the observation the requested number of periods earlier.
    """

    result = calculate_annualized_inflation(
        data=sample_inflation_data(),
        periods=2,
    )

    assert result.iloc[:2].isna().all()

    assert result.iloc[2] == pytest.approx(
        (204.0 / 200.0) - 1
    )

    assert result.iloc[3] == pytest.approx(
        (206.0 / 202.0) - 1
    )

    assert result.iloc[5] == pytest.approx(
        (210.0 / 206.0) - 1
    )


def test_calculate_annualized_inflation_name() -> None:
    """
    The period-inflation Series should have a descriptive
    name.
    """

    result = calculate_annualized_inflation(
        data=sample_inflation_data(),
        periods=2,
    )

    assert (
        result.name
        == "CPI 2-Period Inflation Rate"
    )


def test_calculate_annualized_inflation_supports_one_period() -> None:
    """
    A one-period inflation calculation should match monthly
    inflation.
    """

    period_result = calculate_annualized_inflation(
        data=sample_inflation_data(),
        periods=1,
    )

    monthly_result = calculate_monthly_inflation(
        data=sample_inflation_data()
    )

    pd.testing.assert_series_equal(
        period_result,
        monthly_result.rename(
            "CPI 1-Period Inflation Rate"
        ),
    )


def test_calculate_annualized_inflation_supports_custom_column() -> None:
    """
    Period inflation should support a custom CPI column name.
    """

    result = calculate_annualized_inflation(
        data=custom_column_data(),
        cpi_column="Consumer Index",
        periods=2,
    )

    assert result.iloc[2] == pytest.approx(
        (102.0 / 100.0) - 1
    )

    assert (
        result.name
        == "Consumer Index 2-Period Inflation Rate"
    )


@pytest.mark.parametrize(
    "invalid_periods",
    [
        1.5,
        "2",
        None,
        True,
        False,
    ],
)
def test_calculate_annualized_inflation_rejects_non_integer_periods(
    invalid_periods: object,
) -> None:
    """
    The number of inflation periods must be an integer.
    """

    with pytest.raises(
        TypeError,
        match="periods must be an integer",
    ):
        calculate_annualized_inflation(
            data=sample_inflation_data(),
            periods=invalid_periods,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "invalid_periods",
    [
        0,
        -1,
        -12,
    ],
)
def test_calculate_annualized_inflation_rejects_periods_below_one(
    invalid_periods: int,
) -> None:
    """
    Inflation periods must be at least one.
    """

    with pytest.raises(
        ValueError,
        match="periods must be at least 1",
    ):
        calculate_annualized_inflation(
            data=sample_inflation_data(),
            periods=invalid_periods,
        )


def test_calculate_annualized_inflation_rejects_short_series() -> None:
    """
    A period calculation requires periods plus one CPI
    observations.
    """

    with pytest.raises(
        ValueError,
        match="At least 7 CPI observations",
    ):
        calculate_annualized_inflation(
            data=sample_inflation_data(),
            periods=6,
        )


def test_default_annualized_inflation_requires_thirteen_months() -> None:
    """
    The default 12-period comparison should require 13 CPI
    observations.
    """

    with pytest.raises(
        ValueError,
        match=(
            f"At least {MONTHS_PER_YEAR + 1} "
            "CPI observations"
        ),
    ):
        calculate_annualized_inflation(
            data=sample_inflation_data()
        )


# ------------------------------------------------------------------
# calculate_inflation_adjustment_factor
# ------------------------------------------------------------------


def test_calculate_inflation_adjustment_factor_returns_series() -> None:
    """
    Inflation adjustment factors should be returned as a
    pandas Series.
    """

    result = calculate_inflation_adjustment_factor(
        data=sample_inflation_data()
    )

    assert isinstance(result, pd.Series)


def test_inflation_adjustment_factor_uses_latest_cpi_by_default() -> None:
    """
    The latest CPI should be used as the default purchasing-
    power base.
    """

    result = calculate_inflation_adjustment_factor(
        data=sample_inflation_data()
    )

    assert result.iloc[0] == pytest.approx(
        210.0 / 200.0
    )

    assert result.iloc[1] == pytest.approx(
        210.0 / 202.0
    )

    assert result.iloc[-1] == pytest.approx(1.0)


def test_inflation_adjustment_factor_uses_requested_base_date() -> None:
    """
    A selected base month should determine the adjustment
    factor numerator.
    """

    result = calculate_inflation_adjustment_factor(
        data=sample_inflation_data(),
        base_date="2024-03-01",
    )

    assert result.iloc[0] == pytest.approx(
        204.0 / 200.0
    )

    assert result.iloc[2] == pytest.approx(1.0)

    assert result.iloc[-1] == pytest.approx(
        204.0 / 210.0
    )


def test_inflation_adjustment_factor_normalizes_base_date() -> None:
    """
    A day within an available month should resolve to that
    month's CPI.
    """

    result = calculate_inflation_adjustment_factor(
        data=sample_inflation_data(),
        base_date="2024-04-25",
    )

    assert result.iloc[3] == pytest.approx(1.0)


def test_inflation_adjustment_factor_name_uses_latest_base() -> None:
    """
    The Series name should identify the default base month.
    """

    result = calculate_inflation_adjustment_factor(
        data=sample_inflation_data()
    )

    assert (
        result.name
        == (
            "Inflation Adjustment Factor "
            "(2024-06 Dollars)"
        )
    )


def test_inflation_adjustment_factor_name_uses_requested_base() -> None:
    """
    The Series name should identify a selected base month.
    """

    result = calculate_inflation_adjustment_factor(
        data=sample_inflation_data(),
        base_date="2024-03-20",
    )

    assert (
        result.name
        == (
            "Inflation Adjustment Factor "
            "(2024-03 Dollars)"
        )
    )


def test_inflation_adjustment_factor_supports_custom_column() -> None:
    """
    Adjustment factors should support a custom CPI column.
    """

    result = calculate_inflation_adjustment_factor(
        data=custom_column_data(),
        cpi_column="Consumer Index",
    )

    assert result.iloc[0] == pytest.approx(
        102.0 / 100.0
    )

    assert result.iloc[-1] == pytest.approx(1.0)


def test_inflation_adjustment_factor_rejects_unavailable_base_date() -> None:
    """
    An unavailable purchasing-power month should fail.
    """

    with pytest.raises(
        ValueError,
        match="requested base date is not available",
    ):
        calculate_inflation_adjustment_factor(
            data=sample_inflation_data(),
            base_date="2025-01-01",
        )


def test_inflation_adjustment_factor_rejects_invalid_cpi() -> None:
    """
    Adjustment factors cannot be calculated from nonpositive
    CPI observations.
    """

    data = sample_inflation_data()

    data.loc[1, "CPI"] = -202.0

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        calculate_inflation_adjustment_factor(
            data=data
        )

# ------------------------------------------------------------------
# Part 2 imports
# ------------------------------------------------------------------

from analytics.inflation import (
    build_inflation_summary,
    calculate_cagr,
    calculate_nominal_monthly_returns,
    calculate_real_monthly_returns,
    calculate_real_price,
    calculate_total_change,
    format_currency,
    format_number,
    format_percent,
    generate_inflation_summary,
    print_inflation_summary,
)


# ------------------------------------------------------------------
# calculate_real_price
# ------------------------------------------------------------------


def test_calculate_real_price_returns_series() -> None:
    """
    Inflation-adjusted prices should be returned as a Series.
    """

    result = calculate_real_price(
        data=sample_inflation_data()
    )

    assert isinstance(result, pd.Series)


def test_calculate_real_price_uses_datetime_index() -> None:
    """
    Real prices should remain indexed by date.
    """

    result = calculate_real_price(
        data=sample_inflation_data()
    )

    assert isinstance(
        result.index,
        pd.DatetimeIndex,
    )


def test_calculate_real_price_uses_latest_base_by_default() -> None:
    """
    Historical prices should be expressed in the purchasing
    power of the latest CPI period by default.
    """

    result = calculate_real_price(
        data=sample_inflation_data()
    )

    assert result.iloc[0] == pytest.approx(
        100.0 * (210.0 / 200.0)
    )

    assert result.iloc[1] == pytest.approx(
        110.0 * (210.0 / 202.0)
    )

    assert result.iloc[-1] == pytest.approx(
        139.15
    )


def test_calculate_real_price_uses_requested_base_date() -> None:
    """
    A requested base period should determine the CPI used to
    convert nominal prices.
    """

    result = calculate_real_price(
        data=sample_inflation_data(),
        base_date="2024-03-01",
    )

    assert result.iloc[0] == pytest.approx(
        100.0 * (204.0 / 200.0)
    )

    assert result.iloc[2] == pytest.approx(
        121.0
    )

    assert result.iloc[-1] == pytest.approx(
        139.15 * (204.0 / 210.0)
    )


def test_calculate_real_price_normalizes_base_date() -> None:
    """
    A day within an available month should resolve to that
    month's CPI.
    """

    result = calculate_real_price(
        data=sample_inflation_data(),
        base_date="2024-04-29",
    )

    assert result.iloc[3] == pytest.approx(
        115.0
    )


def test_calculate_real_price_name_uses_latest_base() -> None:
    """
    The real-price Series name should identify the latest
    purchasing-power base month.
    """

    result = calculate_real_price(
        data=sample_inflation_data()
    )

    assert (
        result.name
        == "Real Gold Price (2024-06 Dollars)"
    )


def test_calculate_real_price_name_uses_requested_base() -> None:
    """
    The Series name should identify a requested base month.
    """

    result = calculate_real_price(
        data=sample_inflation_data(),
        base_date="2024-02-20",
    )

    assert (
        result.name
        == "Real Gold Price (2024-02 Dollars)"
    )


def test_calculate_real_price_supports_custom_columns() -> None:
    """
    Real-price calculations should support custom price and
    CPI column names.
    """

    result = calculate_real_price(
        data=custom_column_data(),
        price_column="Silver Price",
        cpi_column="Consumer Index",
    )

    assert result.iloc[0] == pytest.approx(
        20.0 * (102.0 / 100.0)
    )

    assert result.iloc[-1] == pytest.approx(
        24.0
    )

    assert (
        result.name
        == "Real Silver Price (2024-03 Dollars)"
    )


def test_calculate_real_price_rejects_unavailable_base_date() -> None:
    """
    A real-price calculation should fail when the requested
    CPI base month is unavailable.
    """

    with pytest.raises(
        ValueError,
        match="requested base date is not available",
    ):
        calculate_real_price(
            data=sample_inflation_data(),
            base_date="2025-01-01",
        )


def test_calculate_real_price_rejects_invalid_cpi() -> None:
    """
    Nonpositive CPI values should prevent a real-price
    calculation.
    """

    data = sample_inflation_data()

    data.loc[3, "CPI"] = 0.0

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        calculate_real_price(data=data)


def test_calculate_real_price_rejects_missing_price_column() -> None:
    """
    A missing nominal-price column should cause a clear error.
    """

    data = sample_inflation_data().drop(
        columns=["Gold Price"]
    )

    with pytest.raises(ValueError):
        calculate_real_price(data=data)


# ------------------------------------------------------------------
# calculate_nominal_monthly_returns
# ------------------------------------------------------------------


def test_calculate_nominal_monthly_returns_returns_series() -> None:
    """
    Nominal monthly returns should be returned as a Series.
    """

    result = calculate_nominal_monthly_returns(
        data=sample_inflation_data()
    )

    assert isinstance(result, pd.Series)


def test_nominal_monthly_returns_first_value_is_missing() -> None:
    """
    The first nominal return should be missing because no
    prior price exists.
    """

    result = calculate_nominal_monthly_returns(
        data=sample_inflation_data()
    )

    assert pd.isna(result.iloc[0])


def test_calculate_nominal_monthly_return_values() -> None:
    """
    Nominal returns should equal percentage changes in the
    unadjusted price series.
    """

    result = calculate_nominal_monthly_returns(
        data=sample_inflation_data()
    )

    assert result.iloc[1] == pytest.approx(
        (110.0 / 100.0) - 1
    )

    assert result.iloc[2] == pytest.approx(
        (121.0 / 110.0) - 1
    )

    assert result.iloc[3] == pytest.approx(
        (115.0 / 121.0) - 1
    )

    assert result.iloc[-1] == pytest.approx(
        (139.15 / 126.5) - 1
    )


def test_nominal_monthly_returns_name() -> None:
    """
    The nominal-return Series should have a descriptive name.
    """

    result = calculate_nominal_monthly_returns(
        data=sample_inflation_data()
    )

    assert (
        result.name
        == "Gold Price Nominal Monthly Return"
    )


def test_nominal_monthly_returns_support_custom_columns() -> None:
    """
    Nominal returns should support custom price and CPI
    columns.
    """

    result = calculate_nominal_monthly_returns(
        data=custom_column_data(),
        price_column="Silver Price",
        cpi_column="Consumer Index",
    )

    assert result.iloc[1] == pytest.approx(
        (22.0 / 20.0) - 1
    )

    assert result.iloc[2] == pytest.approx(
        (24.0 / 22.0) - 1
    )

    assert (
        result.name
        == "Silver Price Nominal Monthly Return"
    )


def test_nominal_monthly_returns_sort_dates() -> None:
    """
    Nominal returns should be calculated chronologically.
    """

    result = calculate_nominal_monthly_returns(
        data=unsorted_inflation_data()
    )

    expected_index = pd.DatetimeIndex(
        [
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
        ]
    )

    pd.testing.assert_index_equal(
        result.index,
        expected_index,
        check_names=False,
    )

    assert result.iloc[1] == pytest.approx(
        (110.0 / 100.0) - 1
    )


def test_nominal_monthly_returns_reject_invalid_cpi() -> None:
    """
    CPI is used for date alignment and must remain valid.
    """

    data = sample_inflation_data()

    data.loc[1, "CPI"] = -202.0

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        calculate_nominal_monthly_returns(
            data=data
        )


# ------------------------------------------------------------------
# calculate_real_monthly_returns
# ------------------------------------------------------------------


def test_calculate_real_monthly_returns_returns_series() -> None:
    """
    Inflation-adjusted monthly returns should be returned as
    a pandas Series.
    """

    result = calculate_real_monthly_returns(
        data=sample_inflation_data()
    )

    assert isinstance(result, pd.Series)


def test_real_monthly_returns_first_value_is_missing() -> None:
    """
    The first real return should be missing because neither a
    nominal return nor inflation rate can yet be calculated.
    """

    result = calculate_real_monthly_returns(
        data=sample_inflation_data()
    )

    assert pd.isna(result.iloc[0])


def test_calculate_real_monthly_return_values() -> None:
    """
    Real returns should remove CPI inflation from nominal
    price growth.
    """

    result = calculate_real_monthly_returns(
        data=sample_inflation_data()
    )

    first_nominal_return = (
        (110.0 / 100.0) - 1
    )

    first_inflation_rate = (
        (202.0 / 200.0) - 1
    )

    expected_first_real_return = (
        (1 + first_nominal_return)
        / (1 + first_inflation_rate)
    ) - 1

    assert result.iloc[1] == pytest.approx(
        expected_first_real_return
    )

    third_nominal_return = (
        (115.0 / 121.0) - 1
    )

    third_inflation_rate = (
        (206.0 / 204.0) - 1
    )

    expected_third_real_return = (
        (1 + third_nominal_return)
        / (1 + third_inflation_rate)
    ) - 1

    assert result.iloc[3] == pytest.approx(
        expected_third_real_return
    )


def test_real_monthly_return_matches_real_price_change() -> None:
    """
    Real-return calculations should match percentage changes
    in the inflation-adjusted price series.
    """

    real_prices = calculate_real_price(
        data=sample_inflation_data()
    )

    expected_returns = real_prices.pct_change(
        fill_method=None
    )

    result = calculate_real_monthly_returns(
        data=sample_inflation_data()
    )

    pd.testing.assert_series_equal(
        result,
        expected_returns.rename(
            "Gold Price Real Monthly Return"
        ),
    )


def test_real_monthly_returns_name() -> None:
    """
    The real-return Series should have a descriptive name.
    """

    result = calculate_real_monthly_returns(
        data=sample_inflation_data()
    )

    assert (
        result.name
        == "Gold Price Real Monthly Return"
    )


def test_real_monthly_returns_support_custom_columns() -> None:
    """
    Real returns should support custom price and CPI columns.
    """

    result = calculate_real_monthly_returns(
        data=custom_column_data(),
        price_column="Silver Price",
        cpi_column="Consumer Index",
    )

    expected = (
        (22.0 / 20.0)
        / (101.0 / 100.0)
    ) - 1

    assert result.iloc[1] == pytest.approx(
        expected
    )

    assert (
        result.name
        == "Silver Price Real Monthly Return"
    )


def test_real_monthly_returns_are_below_nominal_when_inflation_positive() -> None:
    """
    Positive inflation should make real growth lower than
    nominal growth for the same period.
    """

    nominal_returns = (
        calculate_nominal_monthly_returns(
            data=sample_inflation_data()
        )
    )

    real_returns = calculate_real_monthly_returns(
        data=sample_inflation_data()
    )

    valid_comparison = pd.DataFrame(
        {
            "nominal": nominal_returns,
            "real": real_returns,
        }
    ).dropna()

    assert (
        valid_comparison["real"]
        < valid_comparison["nominal"]
    ).all()


def test_real_monthly_returns_equal_nominal_with_constant_cpi() -> None:
    """
    When CPI does not change, nominal and real returns should
    be equal.
    """

    data = sample_inflation_data()

    data["CPI"] = 200.0

    nominal_returns = (
        calculate_nominal_monthly_returns(
            data=data
        )
    )

    real_returns = calculate_real_monthly_returns(
        data=data
    )

    pd.testing.assert_series_equal(
        real_returns,
        nominal_returns.rename(
            "Gold Price Real Monthly Return"
        ),
    )


def test_real_monthly_returns_reject_invalid_cpi() -> None:
    """
    Real returns cannot be calculated using nonpositive CPI
    values.
    """

    data = sample_inflation_data()

    data.loc[4, "CPI"] = 0.0

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        calculate_real_monthly_returns(
            data=data
        )


# ------------------------------------------------------------------
# calculate_total_change
# ------------------------------------------------------------------


def test_calculate_total_change_for_growth() -> None:
    """
    Total change should measure growth from the first valid
    observation to the last valid observation.
    """

    series = pd.Series(
        [
            100.0,
            110.0,
            121.0,
        ]
    )

    result = calculate_total_change(
        series=series,
        metric_name="test value",
    )

    assert result == pytest.approx(
        (121.0 / 100.0) - 1
    )


def test_calculate_total_change_for_decline() -> None:
    """
    Total change should return a negative result when values
    decline.
    """

    series = pd.Series(
        [
            100.0,
            90.0,
            80.0,
        ]
    )

    result = calculate_total_change(
        series=series,
        metric_name="test value",
    )

    assert result == pytest.approx(-0.20)


def test_calculate_total_change_ignores_missing_values() -> None:
    """
    Missing observations should be removed before selecting
    the first and last valid values.
    """

    series = pd.Series(
        [
            float("nan"),
            100.0,
            float("nan"),
            125.0,
            float("nan"),
        ]
    )

    result = calculate_total_change(
        series=series,
        metric_name="test value",
    )

    assert result == pytest.approx(0.25)


def test_calculate_total_change_allows_negative_starting_value() -> None:
    """
    The helper currently permits negative nonzero values.
    """

    series = pd.Series(
        [
            -100.0,
            -50.0,
        ]
    )

    result = calculate_total_change(
        series=series,
        metric_name="test value",
    )

    assert result == pytest.approx(-0.50)


def test_calculate_total_change_rejects_one_observation() -> None:
    """
    At least two valid observations should be required.
    """

    series = pd.Series(
        [
            float("nan"),
            100.0,
            float("nan"),
        ]
    )

    with pytest.raises(
        ValueError,
        match="At least two test value observations",
    ):
        calculate_total_change(
            series=series,
            metric_name="test value",
        )


def test_calculate_total_change_rejects_empty_series() -> None:
    """
    An empty Series should not produce a total change.
    """

    series = pd.Series(
        dtype="float64"
    )

    with pytest.raises(
        ValueError,
        match="At least two test value observations",
    ):
        calculate_total_change(
            series=series,
            metric_name="test value",
        )


def test_calculate_total_change_rejects_zero_start() -> None:
    """
    A zero starting value would make percentage change
    division invalid.
    """

    series = pd.Series(
        [
            0.0,
            100.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="starting test value value cannot be zero",
    ):
        calculate_total_change(
            series=series,
            metric_name="test value",
        )


# ------------------------------------------------------------------
# calculate_cagr
# ------------------------------------------------------------------


def test_calculate_cagr_for_one_year() -> None:
    """
    A value doubling over approximately one year should
    produce approximately 100% CAGR.
    """

    series = pd.Series(
        [
            100.0,
            200.0,
        ],
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2025-01-01",
            ]
        ),
    )

    elapsed_years = 366 / 365.2425

    expected = (
        (200.0 / 100.0)
        ** (1 / elapsed_years)
    ) - 1

    result = calculate_cagr(
        series=series,
        metric_name="test value",
    )

    assert result == pytest.approx(expected)


def test_calculate_cagr_for_multiple_years() -> None:
    """
    CAGR should compound growth over the elapsed number of
    years.
    """

    series = pd.Series(
        [
            100.0,
            121.0,
        ],
        index=pd.to_datetime(
            [
                "2020-01-01",
                "2022-01-01",
            ]
        ),
    )

    elapsed_days = (
        pd.Timestamp("2022-01-01")
        - pd.Timestamp("2020-01-01")
    ).days

    elapsed_years = (
        elapsed_days / 365.2425
    )

    expected = (
        (121.0 / 100.0)
        ** (1 / elapsed_years)
    ) - 1

    result = calculate_cagr(
        series=series,
        metric_name="test value",
    )

    assert result == pytest.approx(expected)


def test_calculate_cagr_for_decline() -> None:
    """
    CAGR should be negative when the ending value is lower
    than the starting value.
    """

    series = pd.Series(
        [
            100.0,
            81.0,
        ],
        index=pd.to_datetime(
            [
                "2020-01-01",
                "2022-01-01",
            ]
        ),
    )

    result = calculate_cagr(
        series=series,
        metric_name="test value",
    )

    assert result < 0


def test_calculate_cagr_ignores_missing_values() -> None:
    """
    Missing observations should be removed before CAGR is
    calculated.
    """

    series = pd.Series(
        [
            float("nan"),
            100.0,
            float("nan"),
            121.0,
            float("nan"),
        ],
        index=pd.to_datetime(
            [
                "2019-01-01",
                "2020-01-01",
                "2021-01-01",
                "2022-01-01",
                "2023-01-01",
            ]
        ),
    )

    valid_series = pd.Series(
        [
            100.0,
            121.0,
        ],
        index=pd.to_datetime(
            [
                "2020-01-01",
                "2022-01-01",
            ]
        ),
    )

    result = calculate_cagr(
        series=series,
        metric_name="test value",
    )

    expected = calculate_cagr(
        series=valid_series,
        metric_name="test value",
    )

    assert result == pytest.approx(expected)


def test_calculate_cagr_rejects_non_datetime_index() -> None:
    """
    CAGR calculations require dates to determine elapsed
    years.
    """

    series = pd.Series(
        [
            100.0,
            121.0,
        ]
    )

    with pytest.raises(
        TypeError,
        match="require a DatetimeIndex",
    ):
        calculate_cagr(
            series=series,
            metric_name="test value",
        )


def test_calculate_cagr_rejects_one_observation() -> None:
    """
    CAGR requires at least two valid observations.
    """

    series = pd.Series(
        [100.0],
        index=pd.to_datetime(
            ["2024-01-01"]
        ),
    )

    with pytest.raises(
        ValueError,
        match="At least two test value observations",
    ):
        calculate_cagr(
            series=series,
            metric_name="test value",
        )


@pytest.mark.parametrize(
    "values",
    [
        [0.0, 100.0],
        [100.0, 0.0],
        [-100.0, 100.0],
        [100.0, -50.0],
    ],
)
def test_calculate_cagr_rejects_nonpositive_endpoint_values(
    values: list[float],
) -> None:
    """
    CAGR requires positive starting and ending values.
    """

    series = pd.Series(
        values,
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2025-01-01",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="must be greater than zero",
    ):
        calculate_cagr(
            series=series,
            metric_name="test value",
        )


def test_calculate_cagr_rejects_same_date() -> None:
    """
    No CAGR can be calculated when no time has elapsed.
    """

    series = pd.Series(
        [
            100.0,
            121.0,
        ],
        index=pd.to_datetime(
            [
                "2024-01-01",
                "2024-01-01",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="ending date must occur after",
    ):
        calculate_cagr(
            series=series,
            metric_name="test value",
        )


def test_calculate_cagr_rejects_reverse_date_order() -> None:
    """
    The ending observation must occur after the starting
    observation.
    """

    series = pd.Series(
        [
            100.0,
            121.0,
        ],
        index=pd.to_datetime(
            [
                "2025-01-01",
                "2024-01-01",
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="ending date must occur after",
    ):
        calculate_cagr(
            series=series,
            metric_name="test value",
        )

# ------------------------------------------------------------------
# build_inflation_summary
# ------------------------------------------------------------------


def test_build_inflation_summary_returns_dictionary() -> None:
    """
    The inflation summary should be returned as a dictionary.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    assert isinstance(result, dict)


def test_build_inflation_summary_contains_expected_keys() -> None:
    """
    The summary should contain all major nominal, inflation,
    and real-growth metrics.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    expected_keys = {
        "price_column",
        "cpi_column",
        "observation_count",
        "start_date",
        "end_date",
        "base_date",
        "base_cpi",
        "starting_nominal_price",
        "ending_nominal_price",
        "starting_real_price",
        "ending_real_price",
        "starting_cpi",
        "ending_cpi",
        "total_nominal_return",
        "total_inflation",
        "total_real_return",
        "nominal_cagr",
        "inflation_cagr",
        "real_cagr",
        "average_monthly_inflation",
        "average_real_monthly_return",
        "latest_monthly_inflation",
        "latest_real_monthly_return",
    }

    assert set(result.keys()) == expected_keys


def test_build_inflation_summary_metadata() -> None:
    """
    Summary metadata should describe the source columns,
    observation count, and date range.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    assert result["price_column"] == "Gold Price"
    assert result["cpi_column"] == "CPI"
    assert result["observation_count"] == 6
    assert result["start_date"] == "2024-01"
    assert result["end_date"] == "2024-06"


def test_build_inflation_summary_uses_latest_base_by_default() -> None:
    """
    The latest aligned CPI observation should be used as the
    default purchasing-power base.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    assert result["base_date"] == "2024-06"
    assert result["base_cpi"] == pytest.approx(210.0)


def test_build_inflation_summary_uses_requested_base_date() -> None:
    """
    A supplied base month should be reflected in the summary.
    """

    result = build_inflation_summary(
        data=sample_inflation_data(),
        base_date="2024-03-20",
    )

    assert result["base_date"] == "2024-03"
    assert result["base_cpi"] == pytest.approx(204.0)


def test_build_inflation_summary_nominal_prices() -> None:
    """
    Starting and ending nominal prices should match the source
    dataset.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    assert result[
        "starting_nominal_price"
    ] == pytest.approx(100.0)

    assert result[
        "ending_nominal_price"
    ] == pytest.approx(139.15)


def test_build_inflation_summary_real_prices() -> None:
    """
    Real prices should be expressed in the purchasing power
    of the selected base period.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    expected_starting_real_price = (
        100.0 * (210.0 / 200.0)
    )

    expected_ending_real_price = (
        139.15 * (210.0 / 210.0)
    )

    assert result[
        "starting_real_price"
    ] == pytest.approx(
        expected_starting_real_price
    )

    assert result[
        "ending_real_price"
    ] == pytest.approx(
        expected_ending_real_price
    )


def test_build_inflation_summary_cpi_values() -> None:
    """
    Starting and ending CPI values should match the aligned
    data.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    assert result["starting_cpi"] == pytest.approx(
        200.0
    )

    assert result["ending_cpi"] == pytest.approx(
        210.0
    )


def test_build_inflation_summary_total_nominal_return() -> None:
    """
    Total nominal return should compare the first and last
    unadjusted prices.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    expected = (
        139.15 / 100.0
    ) - 1

    assert result[
        "total_nominal_return"
    ] == pytest.approx(expected)


def test_build_inflation_summary_total_inflation() -> None:
    """
    Total inflation should compare the first and last CPI
    observations.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    expected = (
        210.0 / 200.0
    ) - 1

    assert result[
        "total_inflation"
    ] == pytest.approx(expected)


def test_build_inflation_summary_total_real_return() -> None:
    """
    Total real return should compare the first and last
    inflation-adjusted prices.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    starting_real_price = (
        100.0 * (210.0 / 200.0)
    )

    ending_real_price = 139.15

    expected = (
        ending_real_price
        / starting_real_price
    ) - 1

    assert result[
        "total_real_return"
    ] == pytest.approx(expected)


def test_build_inflation_summary_real_return_is_below_nominal() -> None:
    """
    Positive inflation should make total real growth lower
    than total nominal growth.
    """

    result = build_inflation_summary(
        data=sample_inflation_data()
    )

    assert (
        result["total_real_return"]
        < result["total_nominal_return"]
    )


def test_build_inflation_summary_cagr_values() -> None:
    """
    Summary CAGR values should match direct helper
    calculations.
    """

    data = sample_inflation_data()

    aligned_data = align_price_and_cpi(
        data=data
    )

    nominal_prices = aligned_data["Gold Price"]
    cpi = aligned_data["CPI"]

    real_prices = calculate_real_price(
        data=data
    )

    result = build_inflation_summary(
        data=data
    )

    expected_nominal_cagr = calculate_cagr(
        series=nominal_prices,
        metric_name="nominal price",
    )

    expected_inflation_cagr = calculate_cagr(
        series=cpi,
        metric_name="CPI",
    )

    expected_real_cagr = calculate_cagr(
        series=real_prices,
        metric_name="real price",
    )

    assert result["nominal_cagr"] == pytest.approx(
        expected_nominal_cagr
    )

    assert result["inflation_cagr"] == pytest.approx(
        expected_inflation_cagr
    )

    assert result["real_cagr"] == pytest.approx(
        expected_real_cagr
    )


def test_build_inflation_summary_monthly_metrics() -> None:
    """
    Average and latest monthly metrics should match direct
    return and inflation calculations.
    """

    data = sample_inflation_data()

    monthly_inflation = (
        calculate_monthly_inflation(
            data=data
        ).dropna()
    )

    real_returns = (
        calculate_real_monthly_returns(
            data=data
        ).dropna()
    )

    result = build_inflation_summary(
        data=data
    )

    assert result[
        "average_monthly_inflation"
    ] == pytest.approx(
        monthly_inflation.mean()
    )

    assert result[
        "latest_monthly_inflation"
    ] == pytest.approx(
        monthly_inflation.iloc[-1]
    )

    assert result[
        "average_real_monthly_return"
    ] == pytest.approx(
        real_returns.mean()
    )

    assert result[
        "latest_real_monthly_return"
    ] == pytest.approx(
        real_returns.iloc[-1]
    )


def test_build_inflation_summary_supports_custom_columns() -> None:
    """
    The summary should support alternate price and CPI column
    names.
    """

    result = build_inflation_summary(
        data=custom_column_data(),
        price_column="Silver Price",
        cpi_column="Consumer Index",
    )

    assert result["price_column"] == "Silver Price"
    assert result["cpi_column"] == "Consumer Index"
    assert result["observation_count"] == 3
    assert result["base_date"] == "2024-03"
    assert result["base_cpi"] == pytest.approx(102.0)

    assert result[
        "starting_nominal_price"
    ] == pytest.approx(20.0)

    assert result[
        "ending_nominal_price"
    ] == pytest.approx(24.0)


def test_build_inflation_summary_sorts_unsorted_data() -> None:
    """
    Summary calculations should use chronological order even
    when the input rows are unsorted.
    """

    result = build_inflation_summary(
        data=unsorted_inflation_data()
    )

    assert result["start_date"] == "2024-01"
    assert result["end_date"] == "2024-03"

    assert result[
        "starting_nominal_price"
    ] == pytest.approx(100.0)

    assert result[
        "ending_nominal_price"
    ] == pytest.approx(121.0)


def test_build_inflation_summary_rejects_one_observation() -> None:
    """
    At least two aligned observations should be required.
    """

    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2024-01-01"]
            ),
            "Gold Price": [100.0],
            "CPI": [200.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least two aligned",
    ):
        build_inflation_summary(
            data=data
        )


def test_build_inflation_summary_rejects_invalid_cpi() -> None:
    """
    Invalid CPI data should prevent summary generation.
    """

    data = sample_inflation_data()

    data.loc[0, "CPI"] = 0.0

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        build_inflation_summary(
            data=data
        )


def test_build_inflation_summary_rejects_unavailable_base_date() -> None:
    """
    An unavailable purchasing-power base month should fail.
    """

    with pytest.raises(
        ValueError,
        match="requested base date is not available",
    ):
        build_inflation_summary(
            data=sample_inflation_data(),
            base_date="2030-01-01",
        )


# ------------------------------------------------------------------
# generate_inflation_summary
# ------------------------------------------------------------------


def test_generate_inflation_summary_loads_master_file(
    tmp_path: Path,
) -> None:
    """
    The public generator should load the master dataset and
    return an inflation summary.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_inflation_data(),
    )

    result = generate_inflation_summary(
        processed_folder=tmp_path,
    )

    assert result["observation_count"] == 6
    assert result["price_column"] == "Gold Price"
    assert result["cpi_column"] == "CPI"
    assert result["base_date"] == "2024-06"


def test_generate_inflation_summary_supports_custom_filename(
    tmp_path: Path,
) -> None:
    """
    The generator should accept a custom master-data filename.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_inflation_data(),
        file_name="custom_master.csv",
    )

    result = generate_inflation_summary(
        processed_folder=tmp_path,
        file_name="custom_master.csv",
    )

    assert result["observation_count"] == 6
    assert result["start_date"] == "2024-01"
    assert result["end_date"] == "2024-06"


def test_generate_inflation_summary_supports_base_date(
    tmp_path: Path,
) -> None:
    """
    The generator should pass a requested base date into the
    summary builder.
    """

    write_master_csv(
        folder=tmp_path,
        data=sample_inflation_data(),
    )

    result = generate_inflation_summary(
        processed_folder=tmp_path,
        base_date="2024-03-15",
    )

    assert result["base_date"] == "2024-03"
    assert result["base_cpi"] == pytest.approx(204.0)


def test_generate_inflation_summary_supports_custom_columns(
    tmp_path: Path,
) -> None:
    """
    Custom price and CPI columns should work through the file
    loading interface.
    """

    write_master_csv(
        folder=tmp_path,
        data=custom_column_data(),
    )

    result = generate_inflation_summary(
        processed_folder=tmp_path,
        price_column="Silver Price",
        cpi_column="Consumer Index",
    )

    assert result["price_column"] == "Silver Price"
    assert result["cpi_column"] == "Consumer Index"
    assert result["observation_count"] == 3


def test_generate_inflation_summary_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """
    A missing master dataset should raise FileNotFoundError.
    """

    with pytest.raises(FileNotFoundError):
        generate_inflation_summary(
            processed_folder=tmp_path,
            file_name="missing.csv",
        )


def test_generate_inflation_summary_rejects_invalid_dataset(
    tmp_path: Path,
) -> None:
    """
    A master file without the required CPI column should fail.
    """

    invalid_data = sample_inflation_data().drop(
        columns=["CPI"]
    )

    write_master_csv(
        folder=tmp_path,
        data=invalid_data,
    )

    with pytest.raises(ValueError):
        generate_inflation_summary(
            processed_folder=tmp_path,
        )


# ------------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------------


def test_format_percent_positive_value() -> None:
    """
    Positive decimals should be formatted as percentages.
    """

    assert format_percent(0.125) == "12.50%"


def test_format_percent_negative_value() -> None:
    """
    Negative decimals should retain their negative sign.
    """

    assert format_percent(-0.075) == "-7.50%"


def test_format_percent_supports_custom_decimal_places() -> None:
    """
    Percentage formatting should support custom precision.
    """

    assert format_percent(
        0.123456,
        decimal_places=3,
    ) == "12.346%"


def test_format_percent_uses_grouping() -> None:
    """
    Large percentage values should include thousands
    separators.
    """

    assert format_percent(12.5) == "1,250.00%"


def test_format_currency_positive_value() -> None:
    """
    Positive values should be formatted as U.S. currency.
    """

    assert format_currency(1234.5) == "$1,234.50"


def test_format_currency_negative_value() -> None:
    """
    Negative currency values should retain their sign.
    """

    assert format_currency(-50.25) == "$-50.25"


def test_format_currency_supports_custom_decimal_places() -> None:
    """
    Currency formatting should support custom precision.
    """

    assert format_currency(
        1234.5678,
        decimal_places=3,
    ) == "$1,234.568"


def test_format_number_default_precision() -> None:
    """
    Plain numbers should use three decimal places by default.
    """

    assert format_number(210.0) == "210.000"


def test_format_number_uses_grouping() -> None:
    """
    Large numbers should include thousands separators.
    """

    assert format_number(12345.6789) == "12,345.679"


def test_format_number_supports_custom_decimal_places() -> None:
    """
    Number formatting should support custom precision.
    """

    assert format_number(
        1234.5678,
        decimal_places=2,
    ) == "1,234.57"


# ------------------------------------------------------------------
# print_inflation_summary
# ------------------------------------------------------------------


def test_print_inflation_summary_contains_title(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Printed output should contain the report title.
    """

    summary = build_inflation_summary(
        data=sample_inflation_data()
    )

    print_inflation_summary(summary)

    output = capsys.readouterr().out

    assert (
        "GOLD VALUE PREDICTOR — "
        "INFLATION-ADJUSTED ANALYSIS"
        in output
    )


def test_print_inflation_summary_contains_sections(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Printed output should contain each major report section.
    """

    summary = build_inflation_summary(
        data=sample_inflation_data()
    )

    print_inflation_summary(summary)

    output = capsys.readouterr().out

    assert "ANALYSIS PERIOD" in output
    assert "PRICE COMPARISON" in output
    assert "TOTAL GROWTH" in output
    assert "ANNUALIZED GROWTH" in output
    assert "MONTHLY METRICS" in output


def test_print_inflation_summary_contains_metadata(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Printed output should include the selected columns, dates,
    base period, and observation count.
    """

    summary = build_inflation_summary(
        data=sample_inflation_data()
    )

    print_inflation_summary(summary)

    output = capsys.readouterr().out

    assert "Gold Price" in output
    assert "CPI" in output
    assert "6" in output
    assert "2024-01 through 2024-06" in output
    assert "2024-06" in output
    assert "210.000" in output


def test_print_inflation_summary_contains_price_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Printed output should display nominal and real price
    values as currency.
    """

    summary = build_inflation_summary(
        data=sample_inflation_data()
    )

    print_inflation_summary(summary)

    output = capsys.readouterr().out

    assert "$100.00" in output
    assert "$139.15" in output
    assert "$105.00" in output


def test_print_inflation_summary_contains_growth_labels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Printed output should label nominal, inflation, and real
    growth metrics.
    """

    summary = build_inflation_summary(
        data=sample_inflation_data()
    )

    print_inflation_summary(summary)

    output = capsys.readouterr().out

    assert "Total nominal return:" in output
    assert "Total CPI inflation:" in output
    assert "Total real return:" in output
    assert "Nominal CAGR:" in output
    assert "Inflation CAGR:" in output
    assert "Real CAGR:" in output


def test_print_inflation_summary_contains_monthly_labels(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """
    Printed output should label the average and latest monthly
    inflation and real-return metrics.
    """

    summary = build_inflation_summary(
        data=sample_inflation_data()
    )

    print_inflation_summary(summary)

    output = capsys.readouterr().out

    assert "Average monthly inflation:" in output
    assert "Latest monthly inflation:" in output
    assert "Average real monthly return:" in output
    assert "Latest real monthly return:" in output


# ------------------------------------------------------------------
# Integration tests
# ------------------------------------------------------------------


def test_constant_cpi_produces_equal_nominal_and_real_growth() -> None:
    """
    When CPI is constant, nominal and real total returns and
    CAGR values should be equal.
    """

    data = sample_inflation_data()

    data["CPI"] = 200.0

    result = build_inflation_summary(
        data=data
    )

    assert result[
        "total_real_return"
    ] == pytest.approx(
        result["total_nominal_return"]
    )

    assert result["real_cagr"] == pytest.approx(
        result["nominal_cagr"]
    )

    assert result[
        "total_inflation"
    ] == pytest.approx(0.0)

    assert result[
        "inflation_cagr"
    ] == pytest.approx(0.0)


def test_real_price_at_base_month_equals_nominal_price() -> None:
    """
    The inflation-adjusted price in the selected base month
    should equal that month's nominal price.
    """

    data = sample_inflation_data()

    result = calculate_real_price(
        data=data,
        base_date="2024-04-01",
    )

    assert result.loc[
        pd.Timestamp("2024-04-01")
    ] == pytest.approx(115.0)


def test_adjustment_factor_at_base_month_equals_one() -> None:
    """
    The adjustment factor for the selected base month should
    always equal one.
    """

    result = calculate_inflation_adjustment_factor(
        data=sample_inflation_data(),
        base_date="2024-05-01",
    )

    assert result.loc[
        pd.Timestamp("2024-05-01")
    ] == pytest.approx(1.0)


def test_total_real_return_matches_real_price_series() -> None:
    """
    Summary total real return should equal direct total change
    calculated from the real-price Series.
    """

    data = sample_inflation_data()

    real_prices = calculate_real_price(
        data=data
    )

    expected = calculate_total_change(
        series=real_prices,
        metric_name="real price",
    )

    result = build_inflation_summary(
        data=data
    )

    assert result[
        "total_real_return"
    ] == pytest.approx(expected)


def test_summary_base_date_changes_real_price_levels_not_returns() -> None:
    """
    Changing the purchasing-power base rescales every real
    price by the same factor, so real return metrics should
    remain unchanged.
    """

    latest_base_summary = build_inflation_summary(
        data=sample_inflation_data()
    )

    earlier_base_summary = build_inflation_summary(
        data=sample_inflation_data(),
        base_date="2024-03-01",
    )

    assert earlier_base_summary[
        "starting_real_price"
    ] != pytest.approx(
        latest_base_summary[
            "starting_real_price"
        ]
    )

    assert earlier_base_summary[
        "ending_real_price"
    ] != pytest.approx(
        latest_base_summary[
            "ending_real_price"
        ]
    )

    assert earlier_base_summary[
        "total_real_return"
    ] == pytest.approx(
        latest_base_summary[
            "total_real_return"
        ]
    )

    assert earlier_base_summary[
        "real_cagr"
    ] == pytest.approx(
        latest_base_summary[
            "real_cagr"
        ]
    )

    assert earlier_base_summary[
        "average_real_monthly_return"
    ] == pytest.approx(
        latest_base_summary[
            "average_real_monthly_return"
        ]
    )