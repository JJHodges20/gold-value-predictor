import pandas as pd

from analytics.returns import prepare_price_series


DEFAULT_PRICE_COLUMN = "Gold Price"
MONTHS_PER_YEAR = 12


def calculate_historical_cagr(
    data: pd.DataFrame,
    years: int,
    column: str = DEFAULT_PRICE_COLUMN,
) -> float:
    """
    Calculate CAGR using the most recent requested number
    of years.

    Args:
        data:
            DataFrame containing monthly price data.

        years:
            Number of historical years to include.

        column:
            Name of the price column.

    Returns:
        Historical CAGR as a decimal.

    Raises:
        TypeError:
            If years is not an integer.

        ValueError:
            If years is not positive or there is not enough
            historical data.
    """

    if not isinstance(years, int):
        raise TypeError(
            "years must be an integer."
        )

    if years <= 0:
        raise ValueError(
            "years must be greater than zero."
        )

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    required_months = years * MONTHS_PER_YEAR

    if len(prices) <= required_months:
        raise ValueError(
            f"At least {required_months + 1} monthly "
            f"observations are required for a "
            f"{years}-year calculation."
        )

    recent_prices = prices.iloc[
        -(required_months + 1):
    ]

    starting_price = float(
        recent_prices.iloc[0]
    )

    ending_price = float(
        recent_prices.iloc[-1]
    )

    if starting_price <= 0:
        raise ValueError(
            "The starting price must be greater than zero."
        )

    cagr = (
        ending_price / starting_price
    ) ** (1 / years) - 1

    return float(cagr)


def project_price_from_growth(
    current_price: float,
    annual_growth_rate: float,
    forecast_years: int,
) -> float:
    """
    Project a future value using compound annual growth.

    Args:
        current_price:
            Starting price.

        annual_growth_rate:
            Expected annual growth as a decimal.

        forecast_years:
            Number of years to project.

    Returns:
        Hypothetical future price.
    """

    if current_price <= 0:
        raise ValueError(
            "current_price must be greater than zero."
        )

    if not isinstance(forecast_years, int):
        raise TypeError(
            "forecast_years must be an integer."
        )

    if forecast_years <= 0:
        raise ValueError(
            "forecast_years must be greater than zero."
        )

    projected_price = current_price * (
        1 + annual_growth_rate
    ) ** forecast_years

    return float(projected_price)


def generate_growth_forecast(
    data: pd.DataFrame,
    historical_years: int,
    forecast_years: int,
    column: str = DEFAULT_PRICE_COLUMN,
) -> dict[str, float | int | str]:
    """
    Generate a CAGR-based hypothetical gold-price forecast.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    current_price = float(
        prices.iloc[-1]
    )

    historical_cagr = calculate_historical_cagr(
        data=data,
        years=historical_years,
        column=column,
    )

    projected_price = project_price_from_growth(
        current_price=current_price,
        annual_growth_rate=historical_cagr,
        forecast_years=forecast_years,
    )

    latest_date = prices.index[-1]
    forecast_date = latest_date + pd.DateOffset(
        years=forecast_years
    )

    return {
        "method": "Historical CAGR",
        "historical_years": historical_years,
        "forecast_years": forecast_years,
        "current_price": current_price,
        "annual_growth_rate": historical_cagr,
        "projected_price": projected_price,
        "latest_date": latest_date.strftime("%Y-%m"),
        "forecast_date": forecast_date.strftime("%Y-%m"),
    }