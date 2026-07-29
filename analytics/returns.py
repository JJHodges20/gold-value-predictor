from pathlib import Path
from typing import Any

import pandas as pd

from analytics.loader import load_master_data
from data_sources.config import PROCESSED_DATA_FOLDER
from data_sources.merger import DATE_COLUMN, MASTER_FILE_NAME


DEFAULT_PRICE_COLUMN = "Gold Price"
MONTHS_PER_YEAR = 12


def prepare_price_series(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> pd.Series:
    """
    Extract and validate a chronological price series.

    Missing values are removed before the series is returned.

    Args:
        data:
            DataFrame containing a Date column and the requested
            numeric value column.

        column:
            Name of the price column to extract.

    Returns:
        A numeric Series indexed by monthly timestamps.

    Raises:
        ValueError:
            If the dataset is empty, the Date column or requested
            column is missing, dates are invalid, duplicate months
            exist, or no usable values remain.

        TypeError:
            If the requested column is not numeric.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
    )

    if data.empty:
        raise ValueError(
            "Cannot prepare a price series from an empty dataset."
        )

    if DATE_COLUMN not in data.columns:
        raise ValueError(
            f"The dataset is missing the {DATE_COLUMN} column."
        )

    if column not in data.columns:
        raise ValueError(
            f"Column was not found in the dataset: {column}"
        )

    if column == DATE_COLUMN:
        raise ValueError(
            "The Date column cannot be used as a price column."
        )

    if not pd.api.types.is_numeric_dtype(data[column]):
        raise TypeError(
            f"Column is not numeric: {column}"
        )

    prepared_data = data[
        [
            DATE_COLUMN,
            column,
        ]
    ].copy()

    parsed_dates = pd.to_datetime(
        prepared_data[DATE_COLUMN],
        errors="coerce",
    )

    invalid_date_count = int(parsed_dates.isna().sum())

    if invalid_date_count > 0:
        raise ValueError(
            f"The dataset contains {invalid_date_count} "
            "invalid date value(s)."
        )

    prepared_data[DATE_COLUMN] = (
        parsed_dates
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    duplicate_mask = prepared_data[
        DATE_COLUMN
    ].duplicated(keep=False)

    if duplicate_mask.any():
        duplicate_dates = (
            prepared_data.loc[
                duplicate_mask,
                DATE_COLUMN,
            ]
            .dt.strftime("%Y-%m")
            .unique()
            .tolist()
        )

        raise ValueError(
            "The dataset contains duplicate monthly dates: "
            + ", ".join(duplicate_dates)
        )

    prepared_data = (
        prepared_data
        .dropna(subset=[column])
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    if prepared_data.empty:
        raise ValueError(
            f"Column contains no usable observations: {column}"
        )

    price_series = prepared_data.set_index(
        DATE_COLUMN
    )[column].astype(float)

    price_series.name = column

    return price_series


def calculate_monthly_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    drop_missing: bool = True,
) -> pd.Series:
    """
    Calculate month-over-month percentage returns.

    Returns are represented as decimal values. For example,
    0.05 represents a 5% return.

    Args:
        data:
            Source dataset.

        column:
            Numeric price column.

        drop_missing:
            Whether to remove the initial missing return produced
            by percentage-change calculation.

    Returns:
        Series of monthly decimal returns.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    returns = prices.pct_change(
        fill_method=None
    )

    returns.name = f"{column} Monthly Return"

    if drop_missing:
        returns = returns.dropna()

    return returns


def calculate_cumulative_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> pd.Series:
    """
    Calculate cumulative growth from the first observation.

    The first observation receives a cumulative return of 0.0.

    Args:
        data:
            Source dataset.

        column:
            Numeric price column.

    Returns:
        Series of cumulative decimal returns.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    cumulative_returns = (
        prices / prices.iloc[0]
    ) - 1

    cumulative_returns.name = (
        f"{column} Cumulative Return"
    )

    return cumulative_returns


def calculate_total_return(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> float:
    """
    Calculate total return from the first usable observation
    to the last usable observation.

    Args:
        data:
            Source dataset.

        column:
            Numeric price column.

    Returns:
        Total return as a decimal.

    Raises:
        ValueError:
            If fewer than two usable observations exist or the
            initial value is zero.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    if len(prices) < 2:
        raise ValueError(
            "At least two observations are required to "
            "calculate total return."
        )

    initial_value = float(prices.iloc[0])
    final_value = float(prices.iloc[-1])

    if initial_value == 0:
        raise ValueError(
            "Total return cannot be calculated when the "
            "initial value is zero."
        )

    return (final_value / initial_value) - 1


def calculate_cagr(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> float:
    """
    Calculate compound annual growth rate.

    The elapsed period is based on the number of calendar months
    between the first and last usable observations.

    Args:
        data:
            Source dataset.

        column:
            Numeric price column.

    Returns:
        CAGR as a decimal.

    Raises:
        ValueError:
            If fewer than two observations exist, the date range
            has no duration, or either boundary value is not
            positive.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    if len(prices) < 2:
        raise ValueError(
            "At least two observations are required to "
            "calculate CAGR."
        )

    start_date = prices.index[0]
    end_date = prices.index[-1]

    elapsed_months = (
        (end_date.year - start_date.year)
        * MONTHS_PER_YEAR
        + end_date.month
        - start_date.month
    )

    if elapsed_months <= 0:
        raise ValueError(
            "CAGR requires observations from more than "
            "one month."
        )

    initial_value = float(prices.iloc[0])
    final_value = float(prices.iloc[-1])

    if initial_value <= 0 or final_value <= 0:
        raise ValueError(
            "CAGR requires positive initial and final values."
        )

    elapsed_years = (
        elapsed_months / MONTHS_PER_YEAR
    )

    return (
        final_value / initial_value
    ) ** (1 / elapsed_years) - 1


def calculate_annual_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> pd.Series:
    """
    Calculate calendar-year returns using the final available
    observation from each year.

    Each annual return compares one year-end value with the
    previous year-end value.

    Args:
        data:
            Source dataset.

        column:
            Numeric price column.

    Returns:
        Series indexed by calendar year.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    annual_prices = prices.resample(
        "YE"
    ).last()

    annual_returns = annual_prices.pct_change(
        fill_method=None
    ).dropna()

    annual_returns.index = (
        annual_returns.index.year
    )

    annual_returns.index.name = "Year"
    annual_returns.name = (
        f"{column} Annual Return"
    )

    return annual_returns


def calculate_year_to_date_return(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    year: int | None = None,
) -> float:
    """
    Calculate return within one calendar year.

    The calculation uses the first and last usable observations
    available during the requested year.

    Args:
        data:
            Source dataset.

        column:
            Numeric price column.

        year:
            Calendar year to analyze. When omitted, the latest
            year in the dataset is used.

    Returns:
        Year-to-date return as a decimal.

    Raises:
        ValueError:
            If the year contains fewer than two usable values or
            its first value is zero.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    selected_year = (
        int(prices.index.max().year)
        if year is None
        else year
    )

    year_prices = prices[
        prices.index.year == selected_year
    ]

    if len(year_prices) < 2:
        raise ValueError(
            f"At least two observations are required for "
            f"year {selected_year}."
        )

    initial_value = float(year_prices.iloc[0])
    final_value = float(year_prices.iloc[-1])

    if initial_value == 0:
        raise ValueError(
            "Year-to-date return cannot be calculated when "
            "the initial value is zero."
        )

    return (final_value / initial_value) - 1


def calculate_average_monthly_return(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> float:
    """
    Calculate the arithmetic mean monthly return.

    Args:
        data:
            Source dataset.

        column:
            Numeric price column.

    Returns:
        Average monthly return as a decimal.
    """

    returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if returns.empty:
        raise ValueError(
            "No monthly returns are available."
        )

    return float(returns.mean())


def calculate_average_annual_return(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> float:
    """
    Calculate the arithmetic mean of calendar-year returns.

    Args:
        data:
            Source dataset.

        column:
            Numeric price column.

    Returns:
        Average annual return as a decimal.
    """

    returns = calculate_annual_returns(
        data=data,
        column=column,
    )

    if returns.empty:
        raise ValueError(
            "No annual returns are available."
        )

    return float(returns.mean())


def calculate_best_month(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> dict[str, Any]:
    """
    Identify the month with the highest return.

    Returns:
        Dictionary containing the month and decimal return.
    """

    returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if returns.empty:
        raise ValueError(
            "No monthly returns are available."
        )

    best_date = returns.idxmax()

    return {
        "date": best_date.strftime("%Y-%m"),
        "return": float(returns.loc[best_date]),
    }


def calculate_worst_month(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> dict[str, Any]:
    """
    Identify the month with the lowest return.

    Returns:
        Dictionary containing the month and decimal return.
    """

    returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if returns.empty:
        raise ValueError(
            "No monthly returns are available."
        )

    worst_date = returns.idxmin()

    return {
        "date": worst_date.strftime("%Y-%m"),
        "return": float(returns.loc[worst_date]),
    }


def build_return_summary(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
) -> dict[str, Any]:
    """
    Build a complete return-analysis summary.

    Args:
        data:
            Source dataset.

        column:
            Numeric price column.

    Returns:
        Dictionary containing price coverage and return metrics.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    best_month = calculate_best_month(
        data=data,
        column=column,
    )

    worst_month = calculate_worst_month(
        data=data,
        column=column,
    )

    return {
        "column": column,
        "observation_count": int(len(prices)),
        "start_date": prices.index[0].strftime(
            "%Y-%m"
        ),
        "end_date": prices.index[-1].strftime(
            "%Y-%m"
        ),
        "starting_value": float(
            prices.iloc[0]
        ),
        "ending_value": float(
            prices.iloc[-1]
        ),
        "total_return": calculate_total_return(
            data=data,
            column=column,
        ),
        "cagr": calculate_cagr(
            data=data,
            column=column,
        ),
        "average_monthly_return": (
            calculate_average_monthly_return(
                data=data,
                column=column,
            )
        ),
        "average_annual_return": (
            calculate_average_annual_return(
                data=data,
                column=column,
            )
        ),
        "best_month": best_month,
        "worst_month": worst_month,
    }


def generate_return_summary(
    column: str = DEFAULT_PRICE_COLUMN,
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> dict[str, Any]:
    """
    Load the master dataset and generate a return summary.

    Args:
        column:
            Numeric price column.

        processed_folder:
            Folder containing the master dataset.

        file_name:
            Master dataset filename.

    Returns:
        Complete return summary.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    return build_return_summary(
        data=data,
        column=column,
    )


def format_percent(
    value: float,
    decimal_places: int = 2,
) -> str:
    """
    Format a decimal return as a percentage.
    """

    return f"{value * 100:,.{decimal_places}f}%"


def format_value(
    value: float,
    decimal_places: int = 2,
) -> str:
    """
    Format a numeric price value for console output.
    """

    return f"{value:,.{decimal_places}f}"


def print_return_summary(
    summary: dict[str, Any],
) -> None:
    """
    Print a readable return-analysis report.
    """

    separator = "=" * 72
    section_separator = "-" * 72

    print(separator)
    print("GOLD VALUE PREDICTOR — RETURN ANALYSIS")
    print(separator)

    print("\nANALYSIS PERIOD")
    print(section_separator)
    print(
        f"Column:               "
        f"{summary['column']}"
    )
    print(
        f"Observations:         "
        f"{summary['observation_count']:,}"
    )
    print(
        f"Date range:           "
        f"{summary['start_date']} through "
        f"{summary['end_date']}"
    )
    print(
        f"Starting value:       "
        f"{format_value(summary['starting_value'])}"
    )
    print(
        f"Ending value:         "
        f"{format_value(summary['ending_value'])}"
    )

    print("\nRETURN METRICS")
    print(section_separator)
    print(
        f"Total return:         "
        f"{format_percent(summary['total_return'])}"
    )
    print(
        f"CAGR:                 "
        f"{format_percent(summary['cagr'])}"
    )
    print(
        f"Average monthly:      "
        f"{format_percent(summary['average_monthly_return'])}"
    )
    print(
        f"Average annual:       "
        f"{format_percent(summary['average_annual_return'])}"
    )

    print("\nBEST AND WORST MONTHS")
    print(section_separator)
    print(
        f"Best month:           "
        f"{summary['best_month']['date']} "
        f"({format_percent(summary['best_month']['return'])})"
    )
    print(
        f"Worst month:          "
        f"{summary['worst_month']['date']} "
        f"({format_percent(summary['worst_month']['return'])})"
    )

    print(f"\n{separator}")


def main() -> None:
    """
    Generate and print return analysis for gold prices.
    """

    try:
        summary = generate_return_summary()
        print_return_summary(summary)

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            f"Unable to generate return analysis: "
            f"{error}"
        )


if __name__ == "__main__":
    main()