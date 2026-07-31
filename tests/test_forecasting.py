import pandas as pd
import pytest

from analytics.forecasting import (
    DEFAULT_FORECAST_COLUMN,
    DEFAULT_FORECAST_SCENARIOS,
    build_forecast_summary,
    generate_forecast_scenarios,
    generate_forecast_series,
    generate_forecast_summary,
    print_forecast_summary,
    project_future_value,
    validate_growth_rate,
    validate_months_per_year,
    validate_years,
)


def create_price_series(
    starting_price: float = 100.0,
    months: int = 24,
) -> pd.Series:
    """
    Create a predictable monthly price Series for tests.
    """

    dates = pd.date_range(
        start="2024-01-01",
        periods=months,
        freq="MS",
    )

    prices = [
        starting_price + month
        for month in range(months)
    ]

    return pd.Series(
        data=prices,
        index=dates,
        name="Gold Price",
        dtype=float,
    )


# ------------------------------------------------------------------
# Validation tests
# ------------------------------------------------------------------

@pytest.mark.parametrize(
    "years",
    [
        1,
        5,
        10,
    ],
)
def test_validate_years_accepts_positive_integers(
    years: int,
) -> None:
    validate_years(years)


@pytest.mark.parametrize(
    "years",
    [
        0,
        -1,
        -10,
    ],
)
def test_validate_years_rejects_values_below_one(
    years: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be at least 1",
    ):
        validate_years(years)


@pytest.mark.parametrize(
    "years",
    [
        1.5,
        "5",
        None,
        True,
    ],
)
def test_validate_years_rejects_nonintegers(
    years: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validate_years(years)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "growth_rate",
    [
        -0.99,
        -0.10,
        0.0,
        0.05,
        1,
    ],
)
def test_validate_growth_rate_accepts_valid_rates(
    growth_rate: float,
) -> None:
    validate_growth_rate(growth_rate)


@pytest.mark.parametrize(
    "growth_rate",
    [
        -1.0,
        -1.5,
        -10.0,
    ],
)
def test_validate_growth_rate_rejects_rates_at_or_below_negative_one(
    growth_rate: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be greater than -1",
    ):
        validate_growth_rate(growth_rate)


@pytest.mark.parametrize(
    "growth_rate",
    [
        "5%",
        None,
        True,
    ],
)
def test_validate_growth_rate_rejects_nonnumeric_values(
    growth_rate: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be numeric",
    ):
        validate_growth_rate(  # type: ignore[arg-type]
            growth_rate
        )


@pytest.mark.parametrize(
    "months_per_year",
    [
        1,
        4,
        12,
    ],
)
def test_validate_months_per_year_accepts_positive_integers(
    months_per_year: int,
) -> None:
    validate_months_per_year(
        months_per_year
    )


@pytest.mark.parametrize(
    "months_per_year",
    [
        0,
        -1,
    ],
)
def test_validate_months_per_year_rejects_nonpositive_values(
    months_per_year: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="must be at least 1",
    ):
        validate_months_per_year(
            months_per_year
        )


@pytest.mark.parametrize(
    "months_per_year",
    [
        12.0,
        "12",
        None,
        False,
    ],
)
def test_validate_months_per_year_rejects_nonintegers(
    months_per_year: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="must be an integer",
    ):
        validate_months_per_year(  # type: ignore[arg-type]
            months_per_year
        )


# ------------------------------------------------------------------
# Core forecast calculation tests
# ------------------------------------------------------------------

def test_project_future_value_uses_compound_growth() -> None:
    result = project_future_value(
        current_price=100.0,
        annual_growth_rate=0.05,
        years=5,
    )

    assert result == pytest.approx(
        127.62815625
    )


def test_project_future_value_supports_negative_growth() -> None:
    result = project_future_value(
        current_price=100.0,
        annual_growth_rate=-0.10,
        years=2,
    )

    assert result == pytest.approx(81.0)


def test_project_future_value_rejects_nonpositive_price() -> None:
    with pytest.raises(
        ValueError,
        match="current_price must be greater than zero",
    ):
        project_future_value(
            current_price=0.0,
            annual_growth_rate=0.05,
            years=5,
        )


def test_generate_forecast_series_returns_expected_shape() -> None:
    prices = create_price_series()

    result = generate_forecast_series(
        price_series=prices,
        annual_growth_rate=0.05,
        years=2,
    )

    assert isinstance(result, pd.Series)
    assert len(result) == 24
    assert result.name == DEFAULT_FORECAST_COLUMN


def test_generate_forecast_series_starts_next_month() -> None:
    prices = create_price_series(
        months=12
    )

    result = generate_forecast_series(
        price_series=prices,
        annual_growth_rate=0.05,
        years=1,
    )

    expected_start = (
        prices.index[-1]
        + pd.offsets.MonthBegin(1)
    )

    assert result.index[0] == expected_start


def test_generate_forecast_series_reaches_annual_target() -> None:
    prices = pd.Series(
        data=[100.0],
        index=pd.to_datetime(
            ["2026-01-01"]
        ),
        name="Gold Price",
    )

    result = generate_forecast_series(
        price_series=prices,
        annual_growth_rate=0.05,
        years=1,
    )

    assert result.iloc[-1] == pytest.approx(
        105.0
    )


def test_generate_forecast_series_preserves_monthly_frequency() -> None:
    prices = create_price_series()

    result = generate_forecast_series(
        price_series=prices,
        annual_growth_rate=0.05,
        years=1,
    )

    expected_dates = pd.date_range(
        start=prices.index[-1]
        + pd.offsets.MonthBegin(1),
        periods=12,
        freq="MS",
    )

    pd.testing.assert_index_equal(
        result.index,
        expected_dates,
    )


def test_generate_forecast_series_rejects_empty_series() -> None:
    empty_series = pd.Series(
        dtype=float
    )

    with pytest.raises(
        ValueError,
        match="price_series cannot be empty",
    ):
        generate_forecast_series(
            price_series=empty_series,
            annual_growth_rate=0.05,
        )


# ------------------------------------------------------------------
# Scenario forecast tests
# ------------------------------------------------------------------

def test_generate_forecast_scenarios_uses_default_scenarios() -> None:
    prices = create_price_series()

    result = generate_forecast_scenarios(
        price_series=prices,
        years=1,
    )

    assert isinstance(result, pd.DataFrame)

    assert list(result.columns) == list(
        DEFAULT_FORECAST_SCENARIOS.keys()
    )

    assert len(result) == 12


def test_generate_forecast_scenarios_orders_end_values_by_growth_rate() -> None:
    prices = create_price_series()

    result = generate_forecast_scenarios(
        price_series=prices,
        years=1,
    )

    ending_values = result.iloc[-1]

    assert (
        ending_values["Conservative"]
        < ending_values["Expected"]
        < ending_values["Optimistic"]
    )


def test_generate_forecast_scenarios_accepts_custom_rates() -> None:
    prices = create_price_series()

    custom_rates = {
        "Low": 0.01,
        "High": 0.10,
    }

    result = generate_forecast_scenarios(
        price_series=prices,
        growth_rates=custom_rates,
        years=1,
    )

    assert list(result.columns) == [
        "Low",
        "High",
    ]


def test_generate_forecast_scenarios_rejects_empty_dictionary() -> None:
    prices = create_price_series()

    with pytest.raises(
        ValueError,
        match="growth_rates cannot be empty",
    ):
        generate_forecast_scenarios(
            price_series=prices,
            growth_rates={},
        )


def test_generate_forecast_scenarios_rejects_empty_name() -> None:
    prices = create_price_series()

    with pytest.raises(
        ValueError,
        match="Scenario names cannot be empty",
    ):
        generate_forecast_scenarios(
            price_series=prices,
            growth_rates={
                "   ": 0.05,
            },
        )


# ------------------------------------------------------------------
# Summary tests
# ------------------------------------------------------------------

def test_build_forecast_summary_returns_expected_fields() -> None:
    prices = create_price_series()

    result = build_forecast_summary(
        price_series=prices,
        annual_growth_rate=0.05,
        years=2,
        scenario_name="Expected",
    )

    expected_fields = {
        "scenario",
        "current_price",
        "annual_growth_rate",
        "forecast_years",
        "forecast_periods",
        "forecast_start_date",
        "forecast_end_date",
        "projected_price",
        "projected_change",
        "projected_percentage_change",
    }

    assert set(result) == expected_fields
    assert result["scenario"] == "Expected"
    assert result["forecast_years"] == 2
    assert result["forecast_periods"] == 24


def test_build_forecast_summary_calculates_total_percentage_change() -> None:
    prices = pd.Series(
        data=[100.0],
        index=pd.to_datetime(
            ["2026-01-01"]
        ),
        name="Gold Price",
    )

    result = build_forecast_summary(
        price_series=prices,
        annual_growth_rate=0.05,
        years=1,
    )

    assert result[
        "projected_percentage_change"
    ] == pytest.approx(0.05)


def test_generate_forecast_summary_returns_one_row_per_scenario() -> None:
    prices = create_price_series()

    result = generate_forecast_summary(
        price_series=prices,
        years=1,
    )

    assert isinstance(result, pd.DataFrame)

    assert list(result.index) == list(
        DEFAULT_FORECAST_SCENARIOS.keys()
    )

    assert result.index.name == "Scenario"


def test_print_forecast_summary_outputs_disclaimer(
    capsys: pytest.CaptureFixture[str],
) -> None:
    prices = create_price_series()

    summary = generate_forecast_summary(
        price_series=prices,
        years=1,
    )

    print_forecast_summary(summary)

    captured = capsys.readouterr()

    assert (
        "GOLD VALUE FORECAST SUMMARY"
        in captured.out
    )

    assert (
        "not guarantees or predictions"
        in captured.out
    )


def test_print_forecast_summary_rejects_empty_frame() -> None:
    with pytest.raises(
        ValueError,
        match="summary_frame cannot be empty",
    ):
        print_forecast_summary(
            pd.DataFrame()
        )