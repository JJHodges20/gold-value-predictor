from math import isclose, sqrt
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


MONTHS_PER_YEAR = 12
DEFAULT_RISK_FREE_RATE = 0.0
DEFAULT_MINIMUM_ACCEPTABLE_RETURN = 0.0
FLOAT_TOLERANCE = 1e-12


def annual_rate_to_monthly_rate(
    annual_rate: float,
) -> float:
    """
    Convert an annual compound rate into its equivalent
    monthly compound rate.

    Args:
        annual_rate:
            Annual rate expressed as a decimal. For example,
            0.05 represents 5%.

    Returns:
        Equivalent monthly compound rate as a decimal.

    Raises:
        ValueError:
            If the annual rate is less than or equal to -100%.
    """

    if annual_rate <= -1:
        raise ValueError(
            "The annual rate must be greater than -100%."
        )

    monthly_rate = (
        (1 + annual_rate) ** (1 / MONTHS_PER_YEAR)
    ) - 1

    return float(monthly_rate)


def calculate_monthly_volatility(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> float:
    """
    Calculate the sample standard deviation of monthly returns.

    Args:
        data:
            Source dataset containing dates and price values.

        column:
            Numeric price column to analyze.

    Returns:
        Monthly volatility as a decimal.

    Raises:
        ValueError:
            If fewer than two monthly returns are available.
    """

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if len(monthly_returns) < 2:
        raise ValueError(
            "At least two monthly returns are required to "
            "calculate monthly volatility."
        )

    volatility = monthly_returns.std(ddof=1)

    return float(volatility)


def calculate_annualized_volatility(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> float:
    """
    Annualize monthly volatility using square-root-of-time
    scaling.

    Args:
        data:
            Source dataset containing dates and price values.

        column:
            Numeric price column to analyze.

    Returns:
        Annualized volatility as a decimal.
    """

    monthly_volatility = calculate_monthly_volatility(
        data=data,
        column=column,
    )

    annualized_volatility = (
        monthly_volatility
        * sqrt(MONTHS_PER_YEAR)
    )

    return float(annualized_volatility)


def calculate_drawdown_series(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> pd.Series:
    """
    Calculate the percentage decline from the running peak.

    A drawdown of -0.20 means the value is 20% below its
    highest observed value up to that date.

    Args:
        data:
            Source dataset containing dates and price values.

        column:
            Numeric price column to analyze.

    Returns:
        Drawdown Series indexed by date.

    Raises:
        ValueError:
            If a running peak is effectively zero.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    running_peak = prices.cummax()

    has_zero_peak = running_peak.apply(
        lambda value: isclose(
            float(value),
            0.0,
            abs_tol=FLOAT_TOLERANCE,
        )
    ).any()

    if has_zero_peak:
        raise ValueError(
            "Drawdown cannot be calculated when a running "
            "peak is zero."
        )

    drawdowns = (
        prices / running_peak
    ) - 1

    drawdowns.name = f"{column} Drawdown"

    return drawdowns


def calculate_max_drawdown(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> dict[str, Any]:
    """
    Calculate the largest peak-to-trough decline.

    Args:
        data:
            Source dataset containing dates and price values.

        column:
            Numeric price column to analyze.

    Returns:
        Dictionary containing:

        - drawdown
        - peak_date
        - trough_date
        - recovery_date
        - peak_value
        - trough_value
        - recovered
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    drawdowns = calculate_drawdown_series(
        data=data,
        column=column,
    )

    trough_date = drawdowns.idxmin()

    maximum_drawdown = float(
        drawdowns.loc[trough_date]
    )

    prices_through_trough = prices.loc[
        :trough_date
    ]

    peak_value = float(
        prices_through_trough.max()
    )

    peak_dates = prices_through_trough[
        prices_through_trough == peak_value
    ].index

    peak_date = peak_dates[-1]

    prices_after_trough = prices.loc[
        prices.index > trough_date
    ]

    recovery_values = prices_after_trough[
        prices_after_trough >= peak_value
    ]

    recovery_date = (
        recovery_values.index[0]
        if not recovery_values.empty
        else None
    )

    return {
        "drawdown": maximum_drawdown,
        "peak_date": peak_date.strftime("%Y-%m"),
        "trough_date": trough_date.strftime("%Y-%m"),
        "recovery_date": (
            recovery_date.strftime("%Y-%m")
            if recovery_date is not None
            else None
        ),
        "peak_value": peak_value,
        "trough_value": float(
            prices.loc[trough_date]
        ),
        "recovered": recovery_date is not None,
    }


def calculate_downside_deviation(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    annual_minimum_acceptable_return: float = (
        DEFAULT_MINIMUM_ACCEPTABLE_RETURN
    ),
    annualize: bool = True,
) -> float:
    """
    Calculate downside deviation relative to a target return.

    Returns above the target contribute zero downside risk.
    Returns below the target contribute their squared
    shortfall.

    Args:
        data:
            Source dataset containing dates and price values.

        column:
            Numeric price column to analyze.

        annual_minimum_acceptable_return:
            Annual target return expressed as a decimal.

        annualize:
            Whether to annualize the monthly downside
            deviation.

    Returns:
        Downside deviation as a decimal.

    Raises:
        ValueError:
            If no monthly returns are available.
    """

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if monthly_returns.empty:
        raise ValueError(
            "At least one monthly return is required to "
            "calculate downside deviation."
        )

    monthly_target = annual_rate_to_monthly_rate(
        annual_minimum_acceptable_return
    )

    return_shortfalls = (
        monthly_returns - monthly_target
    ).clip(upper=0)

    mean_squared_shortfall = float(
        (return_shortfalls ** 2).mean()
    )

    monthly_downside_deviation = sqrt(
        mean_squared_shortfall
    )

    if annualize:
        annualized_downside_deviation = (
            monthly_downside_deviation
            * sqrt(MONTHS_PER_YEAR)
        )

        return float(
            annualized_downside_deviation
        )

    return float(monthly_downside_deviation)


def calculate_sharpe_ratio(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    annual_risk_free_rate: float = (
        DEFAULT_RISK_FREE_RATE
    ),
) -> float:
    """
    Calculate the annualized Sharpe ratio.

    The ratio compares average excess monthly return with
    monthly excess-return volatility.

    Args:
        data:
            Source dataset containing dates and price values.

        column:
            Numeric price column to analyze.

        annual_risk_free_rate:
            Annual risk-free rate expressed as a decimal.

    Returns:
        Annualized Sharpe ratio.

    Raises:
        ValueError:
            If fewer than two monthly returns are available
            or volatility is effectively zero.
    """

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if len(monthly_returns) < 2:
        raise ValueError(
            "At least two monthly returns are required to "
            "calculate the Sharpe ratio."
        )

    monthly_risk_free_rate = (
        annual_rate_to_monthly_rate(
            annual_risk_free_rate
        )
    )

    excess_returns = (
        monthly_returns
        - monthly_risk_free_rate
    )

    monthly_volatility = float(
        excess_returns.std(ddof=1)
    )

    if isclose(
        monthly_volatility,
        0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError(
            "The Sharpe ratio is undefined when return "
            "volatility is zero."
        )

    average_excess_return = float(
        excess_returns.mean()
    )

    sharpe_ratio = (
        average_excess_return
        / monthly_volatility
    ) * sqrt(MONTHS_PER_YEAR)

    return float(sharpe_ratio)


def calculate_sortino_ratio(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    annual_minimum_acceptable_return: float = (
        DEFAULT_MINIMUM_ACCEPTABLE_RETURN
    ),
) -> float:
    """
    Calculate the annualized Sortino ratio.

    Unlike the Sharpe ratio, the Sortino ratio penalizes only
    returns below the minimum acceptable return.

    Args:
        data:
            Source dataset containing dates and price values.

        column:
            Numeric price column to analyze.

        annual_minimum_acceptable_return:
            Annual target return expressed as a decimal.

    Returns:
        Annualized Sortino ratio.

    Raises:
        ValueError:
            If no monthly returns are available or downside
            deviation is effectively zero.
    """

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if monthly_returns.empty:
        raise ValueError(
            "At least one monthly return is required to "
            "calculate the Sortino ratio."
        )

    monthly_target = annual_rate_to_monthly_rate(
        annual_minimum_acceptable_return
    )

    excess_returns = (
        monthly_returns
        - monthly_target
    )

    shortfalls = excess_returns.clip(
        upper=0
    )

    mean_squared_shortfall = float(
        (shortfalls ** 2).mean()
    )

    monthly_downside_deviation = sqrt(
        mean_squared_shortfall
    )

    if isclose(
        monthly_downside_deviation,
        0.0,
        abs_tol=FLOAT_TOLERANCE,
    ):
        raise ValueError(
            "The Sortino ratio is undefined when downside "
            "deviation is zero."
        )

    average_excess_return = float(
        excess_returns.mean()
    )

    sortino_ratio = (
        average_excess_return
        / monthly_downside_deviation
    ) * sqrt(MONTHS_PER_YEAR)

    return float(sortino_ratio)


def build_risk_summary(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    annual_risk_free_rate: float = (
        DEFAULT_RISK_FREE_RATE
    ),
    annual_minimum_acceptable_return: float = (
        DEFAULT_MINIMUM_ACCEPTABLE_RETURN
    ),
) -> dict[str, Any]:
    """
    Build a complete risk-analysis summary.

    Args:
        data:
            Source dataset containing dates and price values.

        column:
            Numeric price column to analyze.

        annual_risk_free_rate:
            Annual risk-free rate used by the Sharpe ratio.

        annual_minimum_acceptable_return:
            Annual target used by downside deviation and the
            Sortino ratio.

    Returns:
        Dictionary containing volatility, drawdown, and
        risk-adjusted performance metrics.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if len(monthly_returns) < 2:
        raise ValueError(
            "At least two monthly returns are required to "
            "build a complete risk summary."
        )

    monthly_volatility = float(
        monthly_returns.std(ddof=1)
    )

    annualized_volatility = (
        monthly_volatility
        * sqrt(MONTHS_PER_YEAR)
    )

    monthly_target = annual_rate_to_monthly_rate(
        annual_minimum_acceptable_return
    )

    target_shortfalls = (
        monthly_returns - monthly_target
    ).clip(upper=0)

    monthly_downside_deviation = sqrt(
        float(
            (target_shortfalls ** 2).mean()
        )
    )

    annualized_downside_deviation = (
        monthly_downside_deviation
        * sqrt(MONTHS_PER_YEAR)
    )

    maximum_drawdown = calculate_max_drawdown(
        data=data,
        column=column,
    )

    sharpe_ratio = calculate_sharpe_ratio(
        data=data,
        column=column,
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
    )

    sortino_ratio = calculate_sortino_ratio(
        data=data,
        column=column,
        annual_minimum_acceptable_return=(
            annual_minimum_acceptable_return
        ),
    )

    return {
        "column": column,
        "observation_count": int(len(prices)),
        "return_count": int(len(monthly_returns)),
        "start_date": prices.index[0].strftime(
            "%Y-%m"
        ),
        "end_date": prices.index[-1].strftime(
            "%Y-%m"
        ),
        "annual_risk_free_rate": float(
            annual_risk_free_rate
        ),
        "annual_minimum_acceptable_return": float(
            annual_minimum_acceptable_return
        ),
        "monthly_volatility": monthly_volatility,
        "annualized_volatility": float(
            annualized_volatility
        ),
        "annualized_downside_deviation": float(
            annualized_downside_deviation
        ),
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "maximum_drawdown": maximum_drawdown,
    }


def generate_risk_summary(
    column: str = DEFAULT_PRICE_COLUMN,
    annual_risk_free_rate: float = (
        DEFAULT_RISK_FREE_RATE
    ),
    annual_minimum_acceptable_return: float = (
        DEFAULT_MINIMUM_ACCEPTABLE_RETURN
    ),
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> dict[str, Any]:
    """
    Load the master dataset and generate a risk summary.

    Args:
        column:
            Numeric price column to analyze.

        annual_risk_free_rate:
            Annual risk-free rate used by the Sharpe ratio.

        annual_minimum_acceptable_return:
            Annual target used by downside deviation and the
            Sortino ratio.

        processed_folder:
            Folder containing the master dataset.

        file_name:
            Master dataset filename.

    Returns:
        Complete risk-analysis summary.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    return build_risk_summary(
        data=data,
        column=column,
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
        annual_minimum_acceptable_return=(
            annual_minimum_acceptable_return
        ),
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


def format_ratio(
    value: float,
    decimal_places: int = 3,
) -> str:
    """
    Format a risk-adjusted performance ratio.
    """

    return f"{value:,.{decimal_places}f}"


def format_value(
    value: float,
    decimal_places: int = 2,
) -> str:
    """
    Format a numeric value for console output.
    """

    return f"{value:,.{decimal_places}f}"


def print_risk_summary(
    summary: dict[str, Any],
) -> None:
    """
    Print a readable risk-analysis report.
    """

    separator = "=" * 72
    section_separator = "-" * 72

    maximum_drawdown = summary[
        "maximum_drawdown"
    ]

    print(separator)
    print("GOLD VALUE PREDICTOR — RISK ANALYSIS")
    print(separator)

    print("\nANALYSIS PERIOD")
    print(section_separator)

    print(
        f"Column:                      "
        f"{summary['column']}"
    )

    print(
        f"Price observations:          "
        f"{summary['observation_count']:,}"
    )

    print(
        f"Monthly returns:             "
        f"{summary['return_count']:,}"
    )

    print(
        f"Date range:                  "
        f"{summary['start_date']} through "
        f"{summary['end_date']}"
    )

    print("\nRISK ASSUMPTIONS")
    print(section_separator)

    print(
        f"Annual risk-free rate:       "
        f"{format_percent(summary['annual_risk_free_rate'])}"
    )

    print(
        f"Minimum acceptable return:   "
        f"{format_percent(summary['annual_minimum_acceptable_return'])}"
    )

    print("\nVOLATILITY AND RISK RATIOS")
    print(section_separator)

    print(
        f"Monthly volatility:          "
        f"{format_percent(summary['monthly_volatility'])}"
    )

    print(
        f"Annualized volatility:       "
        f"{format_percent(summary['annualized_volatility'])}"
    )

    print(
        f"Annual downside deviation:   "
        f"{format_percent(summary['annualized_downside_deviation'])}"
    )

    print(
        f"Sharpe ratio:                "
        f"{format_ratio(summary['sharpe_ratio'])}"
    )

    print(
        f"Sortino ratio:               "
        f"{format_ratio(summary['sortino_ratio'])}"
    )

    print("\nMAXIMUM DRAWDOWN")
    print(section_separator)

    print(
        f"Maximum drawdown:            "
        f"{format_percent(maximum_drawdown['drawdown'])}"
    )

    print(
        f"Peak date:                   "
        f"{maximum_drawdown['peak_date']}"
    )

    print(
        f"Peak value:                  "
        f"{format_value(maximum_drawdown['peak_value'])}"
    )

    print(
        f"Trough date:                 "
        f"{maximum_drawdown['trough_date']}"
    )

    print(
        f"Trough value:                "
        f"{format_value(maximum_drawdown['trough_value'])}"
    )

    print(
        f"Recovered:                   "
        f"{'Yes' if maximum_drawdown['recovered'] else 'No'}"
    )

    print(
        f"Recovery date:               "
        f"{maximum_drawdown['recovery_date'] or 'N/A'}"
    )

    print(f"\n{separator}")


def main() -> None:
    """
    Generate and print risk analysis for gold prices.
    """

    try:
        summary = generate_risk_summary()
        print_risk_summary(summary)

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            "Unable to generate risk analysis: "
            f"{error}"
        )


if __name__ == "__main__":
    main()