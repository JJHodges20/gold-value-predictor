from pathlib import Path
from typing import Any

import pandas as pd

from analytics.loader import load_master_data
from analytics.returns import (
    DEFAULT_PRICE_COLUMN,
    prepare_price_series,
)
from data_sources.config import PROCESSED_DATA_FOLDER
from data_sources.merger import MASTER_FILE_NAME


DEFAULT_CPI_COLUMN = "CPI"
MONTHS_PER_YEAR = 12


def validate_base_date(
    base_date: str | pd.Timestamp | None,
) -> pd.Timestamp | None:
    """
    Validate and normalize an optional inflation base date.

    The base date determines the purchasing-power period into
    which historical prices are converted.

    Args:
        base_date:
            Date representing the desired purchasing-power
            period. When omitted, the latest available CPI
            observation is used.

    Returns:
        A normalized pandas Timestamp or None.

    Raises:
        TypeError:
            If the base date is not a string, Timestamp,
            or None.

        ValueError:
            If the supplied date cannot be parsed.
    """

    if base_date is None:
        return None

    if not isinstance(
        base_date,
        (str, pd.Timestamp),
    ):
        raise TypeError(
            "The base date must be a date string, "
            "pandas Timestamp, or None."
        )

    try:
        parsed_date = pd.to_datetime(
            base_date,
            errors="raise",
        )

    except (
        TypeError,
        ValueError,
    ) as error:
        raise ValueError(
            f"Unable to parse base date: {base_date}"
        ) from error

    return pd.Timestamp(parsed_date).to_period(
        "M"
    ).to_timestamp()


def prepare_cpi_series(
    data: pd.DataFrame,
    column: str = DEFAULT_CPI_COLUMN,
) -> pd.Series:
    """
    Prepare a clean CPI Series indexed by date.

    Args:
        data:
            Source dataset containing Date and CPI columns.

        column:
            Name of the CPI column.

    Returns:
        A numeric CPI Series with a DatetimeIndex.

    Raises:
        TypeError:
            If data is not a DataFrame.

        ValueError:
            If required columns are missing, dates are invalid,
            CPI values are missing, or CPI values are not
            positive.
    """

    cpi = prepare_price_series(
        data=data,
        column=column,
    )

    if cpi.empty:
        raise ValueError(
            "The CPI Series does not contain any observations."
        )

    if cpi.isna().any():
        raise ValueError(
            "The CPI column contains missing values."
        )

    if (cpi <= 0).any():
        raise ValueError(
            "CPI values must be greater than zero."
        )

    cpi.name = column

    return cpi


def align_price_and_cpi(
    data: pd.DataFrame,
    price_column: str = DEFAULT_PRICE_COLUMN,
    cpi_column: str = DEFAULT_CPI_COLUMN,
) -> pd.DataFrame:
    """
    Align gold prices and CPI values by date.

    Only dates containing both a valid price and a valid CPI
    observation are retained.

    Args:
        data:
            Source dataset containing dates, prices, and CPI.

        price_column:
            Name of the nominal price column.

        cpi_column:
            Name of the CPI column.

    Returns:
        DataFrame indexed by date with aligned price and CPI
        values.

    Raises:
        ValueError:
            If no overlapping observations exist.
    """

    prices = prepare_price_series(
        data=data,
        column=price_column,
    )

    cpi = prepare_cpi_series(
        data=data,
        column=cpi_column,
    )

    aligned_data = pd.concat(
        [
            prices.rename(price_column),
            cpi.rename(cpi_column),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned_data.empty:
        raise ValueError(
            "No overlapping price and CPI observations "
            "are available."
        )

    aligned_data = aligned_data.sort_index()

    return aligned_data


def resolve_base_cpi(
    cpi: pd.Series,
    base_date: str | pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, float]:
    """
    Resolve the CPI value used as the inflation-adjustment base.

    When no base date is supplied, the final CPI observation
    is used.

    Args:
        cpi:
            Prepared CPI Series indexed by date.

        base_date:
            Optional purchasing-power base period.

    Returns:
        Tuple containing the resolved base date and CPI value.

    Raises:
        ValueError:
            If the requested base month is unavailable.
    """

    normalized_base_date = validate_base_date(
        base_date
    )

    if normalized_base_date is None:
        resolved_date = pd.Timestamp(cpi.index[-1])
        resolved_cpi = float(cpi.iloc[-1])

        return resolved_date, resolved_cpi

    normalized_index = pd.DatetimeIndex(
        cpi.index
    ).to_period("M").to_timestamp()

    matching_positions = (
        normalized_index == normalized_base_date
    )

    if not matching_positions.any():
        raise ValueError(
            "The requested base date is not available "
            "in the CPI data: "
            f"{normalized_base_date.strftime('%Y-%m')}."
        )

    matching_cpi = cpi.loc[matching_positions]

    resolved_date = pd.Timestamp(
        matching_cpi.index[-1]
    )

    resolved_cpi = float(
        matching_cpi.iloc[-1]
    )

    return resolved_date, resolved_cpi


def calculate_monthly_inflation(
    data: pd.DataFrame,
    cpi_column: str = DEFAULT_CPI_COLUMN,
) -> pd.Series:
    """
    Calculate month-over-month inflation from CPI values.

    A result of 0.01 represents 1% inflation compared with
    the previous observation.

    Args:
        data:
            Source dataset containing dates and CPI values.

        cpi_column:
            Name of the CPI column.

    Returns:
        Monthly inflation Series indexed by date.
    """

    cpi = prepare_cpi_series(
        data=data,
        column=cpi_column,
    )

    monthly_inflation = cpi.pct_change(
        fill_method=None,
    )

    monthly_inflation.name = (
        f"{cpi_column} Monthly Inflation Rate"
    )

    return monthly_inflation


def calculate_annualized_inflation(
    data: pd.DataFrame,
    cpi_column: str = DEFAULT_CPI_COLUMN,
    periods: int = MONTHS_PER_YEAR,
) -> pd.Series:
    """
    Calculate inflation over a specified number of periods.

    With monthly data and the default 12-period window, this
    represents year-over-year inflation.

    Args:
        data:
            Source dataset containing dates and CPI values.

        cpi_column:
            Name of the CPI column.

        periods:
            Number of CPI observations used in the comparison.

    Returns:
        Period-over-period inflation Series indexed by date.

    Raises:
        TypeError:
            If periods is not an integer.

        ValueError:
            If periods is below one or insufficient CPI
            observations are available.
    """

    if isinstance(periods, bool) or not isinstance(
        periods,
        int,
    ):
        raise TypeError(
            "Inflation periods must be an integer."
        )

    if periods < 1:
        raise ValueError(
            "Inflation periods must be at least 1."
        )

    cpi = prepare_cpi_series(
        data=data,
        column=cpi_column,
    )

    required_observations = periods + 1

    if len(cpi) < required_observations:
        raise ValueError(
            f"At least {required_observations} CPI "
            "observations are required."
        )

    annualized_inflation = cpi.pct_change(
        periods=periods,
        fill_method=None,
    )

    annualized_inflation.name = (
        f"{cpi_column} {periods}-Period Inflation Rate"
    )

    return annualized_inflation


def calculate_inflation_adjustment_factor(
    data: pd.DataFrame,
    cpi_column: str = DEFAULT_CPI_COLUMN,
    base_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """
    Calculate the factor used to convert historical values
    into the purchasing power of a selected base period.

    Formula:

        adjustment factor = base CPI / historical CPI

    Args:
        data:
            Source dataset containing dates and CPI values.

        cpi_column:
            Name of the CPI column.

        base_date:
            Optional purchasing-power base period. The latest
            available CPI observation is used by default.

    Returns:
        Inflation-adjustment factor indexed by date.
    """

    cpi = prepare_cpi_series(
        data=data,
        column=cpi_column,
    )

    resolved_date, base_cpi = resolve_base_cpi(
        cpi=cpi,
        base_date=base_date,
    )

    adjustment_factor = base_cpi / cpi

    base_label = resolved_date.strftime("%Y-%m")

    adjustment_factor.name = (
        f"Inflation Adjustment Factor "
        f"({base_label} Dollars)"
    )

    return adjustment_factor


def calculate_real_price(
    data: pd.DataFrame,
    price_column: str = DEFAULT_PRICE_COLUMN,
    cpi_column: str = DEFAULT_CPI_COLUMN,
    base_date: str | pd.Timestamp | None = None,
) -> pd.Series:
    """
    Convert nominal prices into inflation-adjusted prices.

    Historical prices are expressed in the purchasing power
    of the selected base period.

    Formula:

        real price =
            nominal price × (base CPI / historical CPI)

    Args:
        data:
            Source dataset containing dates, prices, and CPI.

        price_column:
            Name of the nominal price column.

        cpi_column:
            Name of the CPI column.

        base_date:
            Optional purchasing-power base period. The latest
            available CPI observation is used by default.

    Returns:
        Inflation-adjusted price Series indexed by date.
    """

    aligned_data = align_price_and_cpi(
        data=data,
        price_column=price_column,
        cpi_column=cpi_column,
    )

    cpi = aligned_data[cpi_column]

    resolved_date, base_cpi = resolve_base_cpi(
        cpi=cpi,
        base_date=base_date,
    )

    real_price = (
        aligned_data[price_column]
        * (base_cpi / cpi)
    )

    base_label = resolved_date.strftime("%Y-%m")

    real_price.name = (
        f"Real {price_column} "
        f"({base_label} Dollars)"
    )

    return real_price


def calculate_nominal_monthly_returns(
    data: pd.DataFrame,
    price_column: str = DEFAULT_PRICE_COLUMN,
    cpi_column: str = DEFAULT_CPI_COLUMN,
) -> pd.Series:
    """
    Calculate nominal monthly returns for aligned price data.

    Args:
        data:
            Source dataset containing dates, prices, and CPI.

        price_column:
            Name of the nominal price column.

        cpi_column:
            Name of the CPI column used for date alignment.

    Returns:
        Nominal monthly return Series.
    """

    aligned_data = align_price_and_cpi(
        data=data,
        price_column=price_column,
        cpi_column=cpi_column,
    )

    nominal_returns = aligned_data[
        price_column
    ].pct_change(
        fill_method=None,
    )

    nominal_returns.name = (
        f"{price_column} Nominal Monthly Return"
    )

    return nominal_returns


def calculate_real_monthly_returns(
    data: pd.DataFrame,
    price_column: str = DEFAULT_PRICE_COLUMN,
    cpi_column: str = DEFAULT_CPI_COLUMN,
) -> pd.Series:
    """
    Calculate inflation-adjusted monthly returns.

    Formula:

        real return =
            ((1 + nominal return)
            / (1 + inflation rate)) - 1

    Args:
        data:
            Source dataset containing dates, prices, and CPI.

        price_column:
            Name of the nominal price column.

        cpi_column:
            Name of the CPI column.

    Returns:
        Real monthly return Series indexed by date.
    """

    aligned_data = align_price_and_cpi(
        data=data,
        price_column=price_column,
        cpi_column=cpi_column,
    )

    nominal_returns = aligned_data[
        price_column
    ].pct_change(
        fill_method=None,
    )

    inflation_rates = aligned_data[
        cpi_column
    ].pct_change(
        fill_method=None,
    )

    real_returns = (
        (1 + nominal_returns)
        / (1 + inflation_rates)
    ) - 1

    real_returns.name = (
        f"{price_column} Real Monthly Return"
    )

    return real_returns


def calculate_total_change(
    series: pd.Series,
    metric_name: str,
) -> float:
    """
    Calculate total percentage change from first to last value.

    Args:
        series:
            Numeric time Series.

        metric_name:
            Human-readable name for error messages.

    Returns:
        Total percentage change as a decimal.

    Raises:
        ValueError:
            If fewer than two observations are available or
            the starting value is zero.
    """

    valid_values = series.dropna()

    if len(valid_values) < 2:
        raise ValueError(
            f"At least two {metric_name} observations "
            "are required."
        )

    starting_value = float(
        valid_values.iloc[0]
    )

    ending_value = float(
        valid_values.iloc[-1]
    )

    if starting_value == 0:
        raise ValueError(
            f"The starting {metric_name} value cannot be zero."
        )

    return (
        ending_value / starting_value
    ) - 1


def calculate_cagr(
    series: pd.Series,
    metric_name: str,
) -> float:
    """
    Calculate compound annual growth rate for a time Series.

    The elapsed time is based on the dates attached to the
    first and last valid observations.

    Args:
        series:
            Numeric Series with a DatetimeIndex.

        metric_name:
            Human-readable name for error messages.

    Returns:
        Compound annual growth rate as a decimal.

    Raises:
        TypeError:
            If the Series does not use a DatetimeIndex.

        ValueError:
            If too few observations exist, values are not
            positive, or no time has elapsed.
    """

    valid_values = series.dropna()

    if len(valid_values) < 2:
        raise ValueError(
            f"At least two {metric_name} observations "
            "are required to calculate CAGR."
        )

    if not isinstance(
        valid_values.index,
        pd.DatetimeIndex,
    ):
        raise TypeError(
            "CAGR calculations require a DatetimeIndex."
        )

    starting_value = float(
        valid_values.iloc[0]
    )

    ending_value = float(
        valid_values.iloc[-1]
    )

    if starting_value <= 0 or ending_value <= 0:
        raise ValueError(
            f"{metric_name} values must be greater than "
            "zero to calculate CAGR."
        )

    start_date = pd.Timestamp(
        valid_values.index[0]
    )

    end_date = pd.Timestamp(
        valid_values.index[-1]
    )

    elapsed_days = (
        end_date - start_date
    ).days

    if elapsed_days <= 0:
        raise ValueError(
            "The ending date must occur after the "
            "starting date."
        )

    elapsed_years = elapsed_days / 365.2425

    return (
        ending_value / starting_value
    ) ** (1 / elapsed_years) - 1


def build_inflation_summary(
    data: pd.DataFrame,
    price_column: str = DEFAULT_PRICE_COLUMN,
    cpi_column: str = DEFAULT_CPI_COLUMN,
    base_date: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """
    Build a summary comparing nominal and real gold growth.

    Args:
        data:
            Source dataset containing dates, prices, and CPI.

        price_column:
            Name of the nominal price column.

        cpi_column:
            Name of the CPI column.

        base_date:
            Optional purchasing-power base period.

    Returns:
        Dictionary containing inflation and real-return
        statistics.
    """

    aligned_data = align_price_and_cpi(
        data=data,
        price_column=price_column,
        cpi_column=cpi_column,
    )

    if len(aligned_data) < 2:
        raise ValueError(
            "At least two aligned price and CPI "
            "observations are required."
        )

    nominal_prices = aligned_data[
        price_column
    ]

    cpi = aligned_data[
        cpi_column
    ]

    resolved_base_date, base_cpi = resolve_base_cpi(
        cpi=cpi,
        base_date=base_date,
    )

    real_prices = (
        nominal_prices
        * (base_cpi / cpi)
    )

    nominal_returns = nominal_prices.pct_change(
        fill_method=None,
    )

    inflation_rates = cpi.pct_change(
        fill_method=None,
    )

    real_returns = (
        (1 + nominal_returns)
        / (1 + inflation_rates)
    ) - 1

    valid_inflation_rates = (
        inflation_rates.dropna()
    )

    valid_real_returns = real_returns.dropna()

    start_date = pd.Timestamp(
        aligned_data.index[0]
    )

    end_date = pd.Timestamp(
        aligned_data.index[-1]
    )

    return {
        "price_column": price_column,
        "cpi_column": cpi_column,
        "observation_count": int(
            len(aligned_data)
        ),
        "start_date": start_date.strftime(
            "%Y-%m"
        ),
        "end_date": end_date.strftime(
            "%Y-%m"
        ),
        "base_date": (
            resolved_base_date.strftime("%Y-%m")
        ),
        "base_cpi": float(base_cpi),
        "starting_nominal_price": float(
            nominal_prices.iloc[0]
        ),
        "ending_nominal_price": float(
            nominal_prices.iloc[-1]
        ),
        "starting_real_price": float(
            real_prices.iloc[0]
        ),
        "ending_real_price": float(
            real_prices.iloc[-1]
        ),
        "starting_cpi": float(
            cpi.iloc[0]
        ),
        "ending_cpi": float(
            cpi.iloc[-1]
        ),
        "total_nominal_return": (
            calculate_total_change(
                series=nominal_prices,
                metric_name="nominal price",
            )
        ),
        "total_inflation": calculate_total_change(
            series=cpi,
            metric_name="CPI",
        ),
        "total_real_return": (
            calculate_total_change(
                series=real_prices,
                metric_name="real price",
            )
        ),
        "nominal_cagr": calculate_cagr(
            series=nominal_prices,
            metric_name="nominal price",
        ),
        "inflation_cagr": calculate_cagr(
            series=cpi,
            metric_name="CPI",
        ),
        "real_cagr": calculate_cagr(
            series=real_prices,
            metric_name="real price",
        ),
        "average_monthly_inflation": float(
            valid_inflation_rates.mean()
        ),
        "average_real_monthly_return": float(
            valid_real_returns.mean()
        ),
        "latest_monthly_inflation": float(
            valid_inflation_rates.iloc[-1]
        ),
        "latest_real_monthly_return": float(
            valid_real_returns.iloc[-1]
        ),
    }


def generate_inflation_summary(
    price_column: str = DEFAULT_PRICE_COLUMN,
    cpi_column: str = DEFAULT_CPI_COLUMN,
    base_date: str | pd.Timestamp | None = None,
    processed_folder: Path = PROCESSED_DATA_FOLDER,
    file_name: str = MASTER_FILE_NAME,
) -> dict[str, Any]:
    """
    Load the master dataset and generate inflation analysis.

    Args:
        price_column:
            Name of the nominal price column.

        cpi_column:
            Name of the CPI column.

        base_date:
            Optional purchasing-power base period.

        processed_folder:
            Folder containing the master dataset.

        file_name:
            Master dataset filename.

    Returns:
        Dictionary containing inflation-adjusted statistics.
    """

    data = load_master_data(
        processed_folder=processed_folder,
        file_name=file_name,
    )

    return build_inflation_summary(
        data=data,
        price_column=price_column,
        cpi_column=cpi_column,
        base_date=base_date,
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


def format_currency(
    value: float,
    decimal_places: int = 2,
) -> str:
    """
    Format a value as U.S. currency.
    """

    return (
        f"${value:,.{decimal_places}f}"
    )


def format_number(
    value: float,
    decimal_places: int = 3,
) -> str:
    """
    Format a numeric value with grouping.
    """

    return (
        f"{value:,.{decimal_places}f}"
    )


def print_inflation_summary(
    summary: dict[str, Any],
) -> None:
    """
    Print a readable nominal-versus-real analysis report.
    """

    separator = "=" * 76
    section_separator = "-" * 76

    print(separator)
    print(
        "GOLD VALUE PREDICTOR — "
        "INFLATION-ADJUSTED ANALYSIS"
    )
    print(separator)

    print("\nANALYSIS PERIOD")
    print(section_separator)

    print(
        f"Price column:                 "
        f"{summary['price_column']}"
    )

    print(
        f"CPI column:                   "
        f"{summary['cpi_column']}"
    )

    print(
        f"Observations:                 "
        f"{summary['observation_count']:,}"
    )

    print(
        f"Date range:                   "
        f"{summary['start_date']} through "
        f"{summary['end_date']}"
    )

    print(
        f"Purchasing-power base:        "
        f"{summary['base_date']}"
    )

    print(
        f"Base CPI:                     "
        f"{format_number(summary['base_cpi'])}"
    )

    print("\nPRICE COMPARISON")
    print(section_separator)

    print(
        f"Starting nominal price:       "
        f"{format_currency(summary['starting_nominal_price'])}"
    )

    print(
        f"Ending nominal price:         "
        f"{format_currency(summary['ending_nominal_price'])}"
    )

    print(
        f"Starting real price:          "
        f"{format_currency(summary['starting_real_price'])}"
    )

    print(
        f"Ending real price:            "
        f"{format_currency(summary['ending_real_price'])}"
    )

    print("\nTOTAL GROWTH")
    print(section_separator)

    print(
        f"Total nominal return:         "
        f"{format_percent(summary['total_nominal_return'])}"
    )

    print(
        f"Total CPI inflation:          "
        f"{format_percent(summary['total_inflation'])}"
    )

    print(
        f"Total real return:            "
        f"{format_percent(summary['total_real_return'])}"
    )

    print("\nANNUALIZED GROWTH")
    print(section_separator)

    print(
        f"Nominal CAGR:                 "
        f"{format_percent(summary['nominal_cagr'])}"
    )

    print(
        f"Inflation CAGR:               "
        f"{format_percent(summary['inflation_cagr'])}"
    )

    print(
        f"Real CAGR:                    "
        f"{format_percent(summary['real_cagr'])}"
    )

    print("\nMONTHLY METRICS")
    print(section_separator)

    print(
        f"Average monthly inflation:    "
        f"{format_percent(summary['average_monthly_inflation'])}"
    )

    print(
        f"Latest monthly inflation:     "
        f"{format_percent(summary['latest_monthly_inflation'])}"
    )

    print(
        f"Average real monthly return:  "
        f"{format_percent(summary['average_real_monthly_return'])}"
    )

    print(
        f"Latest real monthly return:   "
        f"{format_percent(summary['latest_real_monthly_return'])}"
    )

    print(f"\n{separator}")


def main() -> None:
    """
    Generate and print inflation-adjusted gold analysis.
    """

    try:
        summary = generate_inflation_summary()
        print_inflation_summary(summary)

    except (
        FileNotFoundError,
        TypeError,
        ValueError,
        OSError,
    ) as error:
        print(
            "Unable to generate inflation analysis: "
            f"{error}"
        )


if __name__ == "__main__":
    main()