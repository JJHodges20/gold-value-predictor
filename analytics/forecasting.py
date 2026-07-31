from numbers import Real

from analytics.returns import (
    DEFAULT_PRICE_COLUMN,
    calculate_cagr,
    prepare_price_series,
)

from numbers import Real

import pandas as pd

from analytics.returns import (
    DEFAULT_PRICE_COLUMN,
    calculate_cagr,
    prepare_price_series,
)

# ------------------------------------------------------------------
# Forecast settings
# ------------------------------------------------------------------

DEFAULT_FORECAST_YEARS = 5

DEFAULT_FORECAST_HISTORY_YEARS = 10

DEFAULT_MONTHS_PER_YEAR = 12

DEFAULT_FORECAST_COLUMN = "Projected Gold Price"

DEFAULT_CONSERVATIVE_GROWTH_RATE = 0.03

DEFAULT_EXPECTED_GROWTH_RATE = 0.05

DEFAULT_OPTIMISTIC_GROWTH_RATE = 0.08

DEFAULT_FORECAST_SCENARIOS = {
    "Conservative": DEFAULT_CONSERVATIVE_GROWTH_RATE,
    "Expected": DEFAULT_EXPECTED_GROWTH_RATE,
    "Optimistic": DEFAULT_OPTIMISTIC_GROWTH_RATE,
}


# ------------------------------------------------------------------
# Forecast validation helpers
# ------------------------------------------------------------------

def validate_years(
    years: int,
    parameter_name: str = "years",
) -> None:
    """
    Validate a number of years.

    Args:
        years:
            Number of years to validate.

        parameter_name:
            Human-readable parameter name used in error
            messages.

    Raises:
        TypeError:
            If years is not an integer.

        ValueError:
            If years is less than one.
    """

    if isinstance(years, bool) or not isinstance(
        years,
        int,
    ):
        raise TypeError(
            f"{parameter_name} must be an integer."
        )

    if years < 1:
        raise ValueError(
            f"{parameter_name} must be at least 1."
        )


def validate_growth_rate(
    growth_rate: float,
    parameter_name: str = "growth_rate",
) -> None:
    """
    Validate an annual growth rate.

    Growth rates are represented as decimal values. For
    example, 0.05 represents annual growth of 5%.

    Negative growth is allowed, but the rate must be greater
    than -1. A rate of -1 would represent a complete 100%
    loss.

    Args:
        growth_rate:
            Annual growth rate expressed as a decimal.

        parameter_name:
            Human-readable parameter name used in error
            messages.

    Raises:
        TypeError:
            If growth_rate is not numeric.

        ValueError:
            If growth_rate is less than or equal to -1.
    """

    if isinstance(growth_rate, bool) or not isinstance(
        growth_rate,
        Real,
    ):
        raise TypeError(
            f"{parameter_name} must be numeric."
        )

    if growth_rate <= -1:
        raise ValueError(
            f"{parameter_name} must be greater than -1."
        )


def validate_months_per_year(
    months_per_year: int,
) -> None:
    """
    Validate the number of forecast periods in one year.

    Args:
        months_per_year:
            Number of monthly periods used per year.

    Raises:
        TypeError:
            If months_per_year is not an integer.

        ValueError:
            If months_per_year is less than one.
    """

    if (
        isinstance(months_per_year, bool)
        or not isinstance(months_per_year, int)
    ):
        raise TypeError(
            "months_per_year must be an integer."
        )

    if months_per_year < 1:
        raise ValueError(
            "months_per_year must be at least 1."
        )

# ------------------------------------------------------------------
# Forecast calculations
# ------------------------------------------------------------------

def project_future_value(
    current_price: float,
    annual_growth_rate: float,
    years: int,
) -> float:
    """
    Calculate the projected future value of an asset using
    compound annual growth.

    Args:
        current_price:
            Current asset price.

        annual_growth_rate:
            Annual growth rate expressed as a decimal.

        years:
            Number of years to project.

    Returns:
        The projected future value.
    """

    validate_growth_rate(
        annual_growth_rate,
        parameter_name="annual_growth_rate",
    )

    validate_years(years)

    if current_price <= 0:
        raise ValueError(
            "current_price must be greater than zero."
        )

    return current_price * (
        (1 + annual_growth_rate) ** years
    )


def generate_forecast_series(
    price_series: pd.Series,
    annual_growth_rate: float,
    years: int = DEFAULT_FORECAST_YEARS,
    months_per_year: int = DEFAULT_MONTHS_PER_YEAR,
) -> pd.Series:
    """
    Generate a monthly forecast Series using
    compound annual growth.

    Args:
        price_series:
            Historical price Series indexed by datetime.

        annual_growth_rate:
            Annual growth rate expressed as a decimal.

        years:
            Forecast horizon.

        months_per_year:
            Number of forecast periods generated
            per year.

    Returns:
        A Pandas Series containing projected prices.
    """

    validate_growth_rate(
        annual_growth_rate,
        parameter_name="annual_growth_rate",
    )

    validate_years(years)

    validate_months_per_year(
        months_per_year,
    )

    if price_series.empty:
        raise ValueError(
            "price_series cannot be empty."
        )

    current_price = float(
        price_series.iloc[-1]
    )

    monthly_growth_rate = (
        (1 + annual_growth_rate)
        ** (1 / months_per_year)
        - 1
    )

    forecast_periods = (
        years * months_per_year
    )

    forecast_dates = pd.date_range(
        start=price_series.index[-1]
        + pd.offsets.MonthBegin(1),
        periods=forecast_periods,
        freq="MS",
    )

    forecast_values = []

    projected_price = current_price

    for _ in range(forecast_periods):
        projected_price *= (
            1 + monthly_growth_rate
        )

        forecast_values.append(
            projected_price
        )

    return pd.Series(
        forecast_values,
        index=forecast_dates,
        name=DEFAULT_FORECAST_COLUMN,
    )

# ------------------------------------------------------------------
# Scenario forecasting
# ------------------------------------------------------------------

def generate_forecast_scenarios(
    price_series: pd.Series,
    growth_rates: dict[str, float] | None = None,
    years: int = DEFAULT_FORECAST_YEARS,
    months_per_year: int = DEFAULT_MONTHS_PER_YEAR,
) -> pd.DataFrame:
    """
    Generate multiple monthly forecast scenarios.

    Each scenario uses a different annual growth-rate
    assumption while sharing the same historical starting
    price and forecast horizon.

    Args:
        price_series:
            Historical price Series indexed by datetime.

        growth_rates:
            Mapping of scenario names to annual growth rates.
            Rates must be expressed as decimals.

            Example:

            {
                "Conservative": 0.03,
                "Expected": 0.05,
                "Optimistic": 0.08,
            }

            When omitted, the default forecast scenarios
            are used.

        years:
            Number of years to forecast.

        months_per_year:
            Number of forecast periods generated per year.

    Returns:
        A DataFrame containing one forecast column for each
        scenario.

    Raises:
        TypeError:
            If growth_rates is not a dictionary, a scenario
            name is not a string, or a growth rate is not
            numeric.

        ValueError:
            If growth_rates is empty, a scenario name is
            empty, or another forecast input is invalid.
    """

    validate_years(years)

    validate_months_per_year(
        months_per_year
    )

    if growth_rates is None:
        growth_rates = (
            DEFAULT_FORECAST_SCENARIOS.copy()
        )

    if not isinstance(growth_rates, dict):
        raise TypeError(
            "growth_rates must be a dictionary."
        )

    if not growth_rates:
        raise ValueError(
            "growth_rates cannot be empty."
        )

    scenario_forecasts = {}

    for scenario_name, growth_rate in (
        growth_rates.items()
    ):
        if not isinstance(scenario_name, str):
            raise TypeError(
                "Each scenario name must be a string."
            )

        cleaned_name = scenario_name.strip()

        if not cleaned_name:
            raise ValueError(
                "Scenario names cannot be empty."
            )

        validate_growth_rate(
            growth_rate,
            parameter_name=(
                f"{cleaned_name} growth rate"
            ),
        )

        forecast_series = (
            generate_forecast_series(
                price_series=price_series,
                annual_growth_rate=growth_rate,
                years=years,
                months_per_year=months_per_year,
            )
        )

        scenario_forecasts[cleaned_name] = (
            forecast_series
        )

    scenario_frame = pd.DataFrame(
        scenario_forecasts
    )

    scenario_frame.index.name = (
        price_series.index.name
    )

    return scenario_frame

# ------------------------------------------------------------------
# Forecast summary helpers
# ------------------------------------------------------------------

def build_forecast_summary(
    price_series: pd.Series,
    annual_growth_rate: float,
    years: int = DEFAULT_FORECAST_YEARS,
    months_per_year: int = DEFAULT_MONTHS_PER_YEAR,
    scenario_name: str = "Forecast",
) -> dict[str, object]:
    """
    Build a summary for one forecast scenario.

    Args:
        price_series:
            Historical price Series indexed by datetime.

        annual_growth_rate:
            Annual growth rate expressed as a decimal.

        years:
            Number of years to forecast.

        months_per_year:
            Number of forecast periods generated per year.

        scenario_name:
            Human-readable name assigned to the forecast.

    Returns:
        A dictionary containing the forecast assumptions,
        dates, starting value, projected value, and total
        projected change.

    Raises:
        TypeError:
            If scenario_name is not a string.

        ValueError:
            If scenario_name is empty or another forecast
            input is invalid.
    """

    if not isinstance(scenario_name, str):
        raise TypeError(
            "scenario_name must be a string."
        )

    cleaned_scenario_name = scenario_name.strip()

    if not cleaned_scenario_name:
        raise ValueError(
            "scenario_name cannot be empty."
        )

    forecast_series = generate_forecast_series(
        price_series=price_series,
        annual_growth_rate=annual_growth_rate,
        years=years,
        months_per_year=months_per_year,
    )

    current_price = float(
        price_series.iloc[-1]
    )

    projected_price = float(
        forecast_series.iloc[-1]
    )

    projected_change = (
        projected_price - current_price
    )

    projected_percentage_change = (
        projected_change / current_price
    )

    return {
        "scenario": cleaned_scenario_name,
        "current_price": current_price,
        "annual_growth_rate": float(
            annual_growth_rate
        ),
        "forecast_years": years,
        "forecast_periods": len(
            forecast_series
        ),
        "forecast_start_date": (
            forecast_series.index[0]
        ),
        "forecast_end_date": (
            forecast_series.index[-1]
        ),
        "projected_price": projected_price,
        "projected_change": projected_change,
        "projected_percentage_change": (
            projected_percentage_change
        ),
    }


def generate_forecast_summary(
    price_series: pd.Series,
    growth_rates: dict[str, float] | None = None,
    years: int = DEFAULT_FORECAST_YEARS,
    months_per_year: int = DEFAULT_MONTHS_PER_YEAR,
) -> pd.DataFrame:
    """
    Generate a summary table for multiple forecast
    scenarios.

    Args:
        price_series:
            Historical price Series indexed by datetime.

        growth_rates:
            Mapping of scenario names to annual growth rates.

            When omitted, the default conservative,
            expected, and optimistic scenarios are used.

        years:
            Number of years to forecast.

        months_per_year:
            Number of forecast periods generated per year.

    Returns:
        A DataFrame containing one row for each forecast
        scenario.
    """

    if growth_rates is None:
        growth_rates = (
            DEFAULT_FORECAST_SCENARIOS.copy()
        )

    if not isinstance(growth_rates, dict):
        raise TypeError(
            "growth_rates must be a dictionary."
        )

    if not growth_rates:
        raise ValueError(
            "growth_rates cannot be empty."
        )

    summaries = []

    for scenario_name, growth_rate in (
        growth_rates.items()
    ):
        summary = build_forecast_summary(
            price_series=price_series,
            annual_growth_rate=growth_rate,
            years=years,
            months_per_year=months_per_year,
            scenario_name=scenario_name,
        )

        summaries.append(summary)

    summary_frame = pd.DataFrame(
        summaries
    )

    summary_frame = summary_frame.set_index(
        "scenario"
    )

    summary_frame.index.name = "Scenario"

    return summary_frame


# ------------------------------------------------------------------
# Forecast console formatting
# ------------------------------------------------------------------

def print_forecast_summary(
    summary_frame: pd.DataFrame,
) -> None:
    """
    Print a readable forecast scenario summary.

    Args:
        summary_frame:
            Forecast summary DataFrame returned by
            generate_forecast_summary().

    Raises:
        TypeError:
            If summary_frame is not a DataFrame.

        ValueError:
            If summary_frame is empty or does not contain
            the required summary columns.
    """

    if not isinstance(
        summary_frame,
        pd.DataFrame,
    ):
        raise TypeError(
            "summary_frame must be a Pandas DataFrame."
        )

    if summary_frame.empty:
        raise ValueError(
            "summary_frame cannot be empty."
        )

    required_columns = {
        "current_price",
        "annual_growth_rate",
        "forecast_years",
        "forecast_end_date",
        "projected_price",
        "projected_change",
        "projected_percentage_change",
    }

    missing_columns = (
        required_columns
        - set(summary_frame.columns)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "summary_frame is missing required columns: "
            f"{missing_text}."
        )

    print()
    print("=" * 72)
    print("GOLD VALUE FORECAST SUMMARY")
    print("=" * 72)

    for scenario_name, row in (
        summary_frame.iterrows()
    ):
        forecast_end_date = (
            pd.Timestamp(
                row["forecast_end_date"]
            ).strftime("%B %Y")
        )

        print()
        print(f"{scenario_name} Scenario")
        print("-" * 72)

        print(
            "Annual Growth Rate: "
            f"{row['annual_growth_rate']:.2%}"
        )

        print(
            "Forecast Horizon: "
            f"{int(row['forecast_years'])} years"
        )

        print(
            "Current Price: "
            f"${row['current_price']:,.2f}"
        )

        print(
            "Projected Price: "
            f"${row['projected_price']:,.2f}"
        )

        print(
            "Projected Change: "
            f"${row['projected_change']:,.2f}"
        )

        print(
            "Projected Percentage Change: "
            f"{row['projected_percentage_change']:.2%}"
        )

        print(
            "Forecast End Date: "
            f"{forecast_end_date}"
        )

    print()
    print("=" * 72)
    print(
        "Forecasts are hypothetical scenarios based on "
        "constant annual growth assumptions."
    )
    print(
        "They are not guarantees or predictions of future "
        "gold prices."
    )
    print("=" * 72)



def main() -> None:
    """
    Preview the forecast calculations and summaries.
    """

    historical_prices = pd.Series(
        data=[
            1800.00,
            1850.00,
            1900.00,
        ],
        index=pd.to_datetime(
            [
                "2026-01-01",
                "2026-02-01",
                "2026-03-01",
            ]
        ),
        name="Price",
    )

    projected_price = project_future_value(
        current_price=100.00,
        annual_growth_rate=0.05,
        years=5,
    )

    scenario_forecasts = (
        generate_forecast_scenarios(
            price_series=historical_prices,
            years=1,
        )
    )

    forecast_summary = (
        generate_forecast_summary(
            price_series=historical_prices,
            years=1,
        )
    )

    print(
        "Five-Year Test Projection: "
        f"${projected_price:.2f}"
    )

    print()
    print("First Five Forecast Rows:")
    print(
        scenario_forecasts.head()
    )

    print_forecast_summary(
        forecast_summary
    )


if __name__ == "__main__":
    main()