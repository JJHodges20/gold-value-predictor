from math import sqrt
from pathlib import Path
from typing import Any

import pandas as pd

from analytics.loader import load_master_data
from analytics.returns import (
    DEFAULT_PRICE_COLUMN,
    calculate_monthly_returns,
    prepare_price_series,
)
from data_sources.config import PROCESSED_DATA_FOLDER
from data_sources.merger import MASTER_FILE_NAME


DEFAULT_ROLLING_WINDOW = 12
MONTHS_PER_YEAR = 12


def validate_window(
    window: int,
) -> None:
    """
    Validate a rolling-window size.

    Args:
        window:
            Number of observations in the rolling window.

    Raises:
        TypeError:
            If the window is not an integer.

        ValueError:
            If the window is less than one.
    """

    if isinstance(window, bool) or not isinstance(
        window,
        int,
    ):
        raise TypeError(
            "The rolling window must be an integer."
        )

    if window < 1:
        raise ValueError(
            "The rolling window must be at least 1."
        )


def validate_price_window(
    prices: pd.Series,
    window: int,
) -> None:
    """
    Confirm that a price Series contains enough observations
    for a requested rolling window.

    Args:
        prices:
            Prepared price Series.

        window:
            Number of price observations in the window.

    Raises:
        ValueError:
            If the Series is shorter than the window.
    """

    if len(prices) < window:
        raise ValueError(
            f"At least {window} price observations are "
            "required for this rolling calculation."
        )


def validate_return_window(
    monthly_returns: pd.Series,
    window: int,
) -> None:
    """
    Confirm that a return Series contains enough observations
    for a requested rolling window.

    Args:
        monthly_returns:
            Monthly return Series.

        window:
            Number of returns in the rolling window.

    Raises:
        ValueError:
            If the Series is shorter than the window.
    """

    if len(monthly_returns) < window:
        raise ValueError(
            f"At least {window} monthly returns are "
            "required for this rolling calculation."
        )


def calculate_rolling_average(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """
    Calculate the rolling arithmetic mean of price values.

    Args:
        data:
            Source dataset containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of price observations in each window.

    Returns:
        Rolling-average Series indexed by date.
    """

    validate_window(window)

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    validate_price_window(
        prices=prices,
        window=window,
    )

    rolling_average = prices.rolling(
        window=window,
        min_periods=window,
    ).mean()

    rolling_average.name = (
        f"{column} {window}-Month Rolling Average"
    )

    return rolling_average


def calculate_rolling_return(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """
    Calculate the percentage change over each rolling period.

    For a 12-month window, each result compares the current
    price with the price 12 observations earlier.

    Args:
        data:
            Source dataset containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of periods across which return is measured.

    Returns:
        Rolling-return Series indexed by date.
    """

    validate_window(window)

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    required_observations = window + 1

    validate_price_window(
        prices=prices,
        window=required_observations,
    )

    rolling_return = prices.pct_change(
        periods=window,
        fill_method=None,
    )

    rolling_return.name = (
        f"{column} {window}-Month Rolling Return"
    )

    return rolling_return


def calculate_rolling_volatility(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    annualize: bool = True,
) -> pd.Series:
    """
    Calculate rolling volatility from monthly returns.

    Volatility is calculated using the sample standard
    deviation. When annualized, monthly volatility is
    multiplied by the square root of 12.

    Args:
        data:
            Source dataset containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly returns in each window.

        annualize:
            Whether to annualize the rolling volatility.

    Returns:
        Rolling-volatility Series indexed by date.
    """

    validate_window(window)

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    validate_return_window(
        monthly_returns=monthly_returns,
        window=window,
    )

    rolling_volatility = monthly_returns.rolling(
        window=window,
        min_periods=window,
    ).std(ddof=1)

    if annualize:
        rolling_volatility = (
            rolling_volatility
            * sqrt(MONTHS_PER_YEAR)
        )

        rolling_volatility.name = (
            f"{column} {window}-Month "
            "Annualized Rolling Volatility"
        )

    else:
        rolling_volatility.name = (
            f"{column} {window}-Month "
            "Rolling Volatility"
        )

    return rolling_volatility


def calculate_rolling_high(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """
    Calculate the highest price within each rolling window.

    Args:
        data:
            Source dataset containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of price observations in each window.

    Returns:
        Rolling-high Series indexed by date.
    """

    validate_window(window)

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    validate_price_window(
        prices=prices,
        window=window,
    )

    rolling_high = prices.rolling(
        window=window,
        min_periods=window,
    ).max()

    rolling_high.name = (
        f"{column} {window}-Month Rolling High"
    )

    return rolling_high


def calculate_rolling_low(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """
    Calculate the lowest price within each rolling window.

    Args:
        data:
            Source dataset containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of price observations in each window.

    Returns:
        Rolling-low Series indexed by date.
    """

    validate_window(window)

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    validate_price_window(
        prices=prices,
        window=window,
    )

    rolling_low = prices.rolling(
        window=window,
        min_periods=window,
    ).min()

    rolling_low.name = (
        f"{column} {window}-Month Rolling Low"
    )

    return rolling_low


def calculate_rolling_drawdown(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """
    Calculate each price's decline from the highest price
    within its rolling window.

    A value of -0.15 means the current price is 15% below
    the highest price observed during the window.

    Args:
        data:
            Source dataset containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of price observations in each window.

    Returns:
        Rolling-drawdown Series indexed by date.
    """

    validate_window(window)

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    validate_price_window(
        prices=prices,
        window=window,
    )

    rolling_high = prices.rolling(
        window=window,
        min_periods=window,
    ).max()

    if (rolling_high == 0).any():
        raise ValueError(
            "Rolling drawdown cannot be calculated when a "
            "rolling high is zero."
        )

    rolling_drawdown = (
        prices / rolling_high
    ) - 1

    rolling_drawdown.name = (
        f"{column} {window}-Month Rolling Drawdown"
    )

    return rolling_drawdown


def calculate_rolling_range(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> pd.Series:
    """
    Calculate the difference between the rolling high and
    rolling low.

    Args:
        data:
            Source dataset containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of price observations in each window.

    Returns:
        Rolling price-range Series indexed by date.
    """

    rolling_high = calculate_rolling_high(
        data=data,
        column=column,
        window=window,
    )

    rolling_low = calculate_rolling_low(
        data=data,
        column=column,
        window=window,
    )

    rolling_range = (
        rolling_high - rolling_low
    )

    rolling_range.name = (
        f"{column} {window}-Month Rolling Range"
    )

    return rolling_range


def get_latest_valid_value(
    series: pd.Series,
    metric_name: str,
) -> float:
    """
    Return the latest non-missing value from a Series.

    Args:
        series:
            Series containing rolling results.

        metric_name:
            Human-readable metric name for error messages.

    Returns:
        Latest valid value as a float.

    Raises:
        ValueError:
            If the Series has no valid values.
    """

    valid_values = series.dropna()

    if valid_values.empty:
        raise ValueError(
            f"No valid {metric_name} values are available."
        )

    return float(valid_values.iloc[-1])


def get_latest_valid_date(
    series: pd.Series,
    metric_name: str,
) -> str:
    """
    Return the date of the latest non-missing Series value.

    Args:
        series:
            Series containing rolling results.

        metric_name:
            Human-readable metric name for error messages.

    Returns:
        Date formatted as YYYY-MM.

    Raises:
        ValueError:
            If the Series has no valid values.
    """

    valid_values = series.dropna()

    if valid_values.empty:
        raise ValueError(
            f"No valid {metric_name} values are available."
        )

    latest_date = valid_values.index[-1]

    return latest_date.strftime("%Y-%m")


def build_rolling_summary(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, Any]:
    """
    Build a summary of the latest rolling statistics.

    Args:
        data:
            Source dataset containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of observations in each rolling window.

    Returns:
        Dictionary containing current rolling statistics.
    """

    validate_window(window)

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    validate_price_window(
        prices=prices,
        window=window + 1,
    )

    rolling_average = calculate_rolling_average(
        data=data,
        column=column,
        window=window,
    )

    rolling_return = calculate_rolling_return(
        data=data,
        column=column,
        window=window,
    )

    rolling_volatility = (
        calculate_rolling_volatility(
            data=data,
            column=column,
            window=window,
            annualize=True,
        )
    )

    rolling_high = calculate_rolling_high(
        data=data,
        column=column,
        window=window,
    )

    rolling_low = calculate_rolling_low(
        data=data,
        column=column,
        window=window,
    )

    rolling_drawdown = calculate_rolling_drawdown(
        data=data,
        column=column,
        window=window,
    )

    rolling_range = calculate_rolling_range(
        data=data,
        column=column,
        window=window,
    )

    latest_date = get_latest_valid_date(
        series=rolling_average,
        metric_name="rolling average",
    )

    return {
        "column": column,
        "window": window,
        "observation_count": int(len(prices)),
        "start_date": prices.index[0].strftime(
            "%Y-%m"
        ),
        "end_date": prices.index[-1].strftime(
            "%Y-%m"
        ),
        "latest_metric_date": latest_date,
        "latest_price": float(prices.iloc[-1]),
        "rolling_average": get_latest_valid_value(
            series=rolling_average,
            metric_name="rolling average",
        ),
        "rolling_return": get_latest_valid_value(
            series=rolling_return,
            metric_name="rolling return",
        ),
        "annualized_rolling_volatility": (
            get_latest_valid_value(
                series=rolling_volatility,
                metric_name="rolling volatility",
            )
        ),
        "rolling_high": get_latest_valid_value(
            series=rolling_high,
            metric_name="rolling high",
        ),
        "rolling_low": get_latest_valid_value(
            series=rolling_low,
            metric_name="rolling low",
        ),
        "rolling_range": get_latest_valid_value(
            series=rolling_range,
            metric_name="rolling range",
        ),
        "rolling_drawdown": (
            get_latest_valid_value(
                series=rolling_drawdown,
                metric_name="rolling drawdown",
            )
        ),
    }


def generate_rolling_summary(
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> dict[str, Any]:
    """
    Load the master dataset and generate rolling statistics.

    Args:
        column:
            Numeric price column to analyze.

        window:
            Number of observations in each rolling window.

        processed_folder:
            Folder containing the master dataset.

        file_name:
            Master dataset filename.

    Returns:
        Dictionary containing current rolling statistics.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    return build_rolling_summary(
        data=data,
        column=column,
        window=window,
    )


def format_percent(
    value: float,
    decimal_places: int = 2,
) -> str:
    """
    Format a decimal value as a percentage.
    """

    return (
        f"{value * 100:,.{decimal_places}f}%"
    )


def format_value(
    value: float,
    decimal_places: int = 2,
) -> str:
    """
    Format a numeric value for console output.
    """

    return f"{value:,.{decimal_places}f}"


def print_rolling_summary(
    summary: dict[str, Any],
) -> None:
    """
    Print a readable rolling-statistics report.
    """

    separator = "=" * 72
    section_separator = "-" * 72

    window = summary["window"]

    print(separator)
    print("GOLD VALUE PREDICTOR — ROLLING ANALYSIS")
    print(separator)

    print("\nANALYSIS PERIOD")
    print(section_separator)

    print(
        f"Column:                      "
        f"{summary['column']}"
    )

    print(
        f"Rolling window:              "
        f"{window} months"
    )

    print(
        f"Price observations:          "
        f"{summary['observation_count']:,}"
    )

    print(
        f"Full date range:             "
        f"{summary['start_date']} through "
        f"{summary['end_date']}"
    )

    print(
        f"Latest metric date:          "
        f"{summary['latest_metric_date']}"
    )

    print("\nLATEST ROLLING METRICS")
    print(section_separator)

    print(
        f"Latest price:                "
        f"{format_value(summary['latest_price'])}"
    )

    print(
        f"{window}-month average:            "
        f"{format_value(summary['rolling_average'])}"
    )

    print(
        f"{window}-month return:             "
        f"{format_percent(summary['rolling_return'])}"
    )

    print(
        f"Annualized volatility:       "
        f"{format_percent(summary['annualized_rolling_volatility'])}"
    )

    print(
        f"{window}-month high:               "
        f"{format_value(summary['rolling_high'])}"
    )

    print(
        f"{window}-month low:                "
        f"{format_value(summary['rolling_low'])}"
    )

    print(
        f"{window}-month range:              "
        f"{format_value(summary['rolling_range'])}"
    )

    print(
        f"Drawdown from rolling high:  "
        f"{format_percent(summary['rolling_drawdown'])}"
    )

    print(f"\n{separator}")


def main() -> None:
    """
    Generate and print rolling analysis for gold prices.
    """

    try:
        summary = generate_rolling_summary()
        print_rolling_summary(summary)

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            "Unable to generate rolling analysis: "
            f"{error}"
        )


if __name__ == "__main__":
    main()