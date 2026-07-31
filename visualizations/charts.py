"""
Reusable chart functions for the Gold Value Predictor.

Each plotting function returns its Matplotlib Figure and Axes
objects instead of displaying the chart automatically. This
allows charts to be tested, customized, displayed, or saved by
other parts of the application.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from analytics.rolling import (
    DEFAULT_ROLLING_WINDOW,
    calculate_rolling_average,
    calculate_rolling_drawdown,
    calculate_rolling_high,
    calculate_rolling_low,
    calculate_rolling_return,
    calculate_rolling_volatility,
)

from analytics.inflation import calculate_real_price

from analytics.returns import (
    DEFAULT_PRICE_COLUMN,
    calculate_annual_returns,
    calculate_cumulative_returns,
    calculate_monthly_returns,
    prepare_price_series,
)

# ------------------------------------------------------------------
# General chart settings
# ------------------------------------------------------------------

DEFAULT_CHART_TITLE = "Historical Gold Prices"

DEFAULT_X_AXIS_LABEL = "Date"

DEFAULT_Y_AXIS_LABEL = "Gold Price (USD)"

DEFAULT_FIGURE_SIZE = (12.0, 6.0)

DEFAULT_LINE_WIDTH = 2.0

DEFAULT_GRID_ALPHA = 0.25

DEFAULT_DPI = 300

DEFAULT_TITLE_PADDING = 22.0

DEFAULT_LABEL_PADDING = 10.0

DEFAULT_ANNOTATION_OFFSET = (12, 12)


# ------------------------------------------------------------------
# Recent-price chart settings
# ------------------------------------------------------------------

DEFAULT_RECENT_YEARS = 5

DEFAULT_RECENT_PRICE_TITLE = "Recent Gold Prices"


# ------------------------------------------------------------------
# Inflation-adjusted chart settings
# ------------------------------------------------------------------

DEFAULT_INFLATION_ADJUSTED_COLUMN = (
    "Inflation_Adjusted_Price"
)

DEFAULT_INFLATION_ADJUSTED_TITLE = (
    "Inflation-Adjusted Gold Prices"
)

DEFAULT_INFLATION_ADJUSTED_LABEL = (
    "Inflation-Adjusted Price"
)

DEFAULT_INFLATION_ADJUSTED_Y_AXIS_LABEL = (
    "Inflation-Adjusted Price (USD per Troy Ounce)"
)

DEFAULT_NOMINAL_VS_REAL_TITLE = (
    "Nominal vs. Inflation-Adjusted Gold Prices"
)

DEFAULT_NOMINAL_LABEL = "Nominal Gold Price"

DEFAULT_REAL_LABEL = (
    "Inflation-Adjusted Gold Price"
)

DEFAULT_COMPARISON_Y_AXIS_LABEL = (
    "Gold Price (USD per Troy Ounce)"
)


# ------------------------------------------------------------------
# Return chart settings
# ------------------------------------------------------------------

DEFAULT_MONTHLY_RETURNS_TITLE = (
    "Monthly Gold Price Returns"
)

DEFAULT_MONTHLY_RETURNS_LABEL = (
    "Monthly Return"
)

DEFAULT_RETURNS_Y_AXIS_LABEL = (
    "Monthly Return"
)

DEFAULT_ZERO_LINE_WIDTH = 1.0

DEFAULT_REFERENCE_LINE_WIDTH = 1.25

DEFAULT_PERCENTAGE_FILL_ALPHA = 0.20


# ------------------------------------------------------------------
# Rolling-analysis chart settings
# ------------------------------------------------------------------

DEFAULT_ROLLING_AVERAGE_TITLE = (
    "Gold Price with Rolling Average"
)

DEFAULT_PRICE_LABEL = "Gold Price"

DEFAULT_ROLLING_AVERAGE_LABEL = (
    "{window}-Month Rolling Average"
)

DEFAULT_ROLLING_Y_AXIS_LABEL = (
    "Gold Price (USD per Troy Ounce)"
)

DEFAULT_ROLLING_VOLATILITY_TITLE = (
    "Rolling Gold Price Volatility"
)

DEFAULT_ROLLING_VOLATILITY_LABEL = (
    "{window}-Month Annualized Volatility"
)

DEFAULT_NON_ANNUALIZED_VOLATILITY_LABEL = (
    "{window}-Month Volatility"
)

DEFAULT_VOLATILITY_Y_AXIS_LABEL = (
    "Annualized Volatility"
)

DEFAULT_NON_ANNUALIZED_VOLATILITY_Y_AXIS_LABEL = (
    "Monthly Volatility"
)

DEFAULT_ROLLING_RETURN_TITLE = (
    "Rolling Gold Price Return"
)

DEFAULT_ROLLING_RETURN_LABEL = (
    "{window}-Month Rolling Return"
)

DEFAULT_ROLLING_RETURN_Y_AXIS_LABEL = (
    "Rolling Return"
)

DEFAULT_ROLLING_DRAWDOWN_TITLE = (
    "Rolling Gold Price Drawdown"
)

DEFAULT_ROLLING_DRAWDOWN_LABEL = (
    "{window}-Month Rolling Drawdown"
)

DEFAULT_ROLLING_DRAWDOWN_Y_AXIS_LABEL = (
    "Drawdown"
)

DEFAULT_ROLLING_HIGH_LOW_TITLE = (
    "Gold Price with Rolling High and Low"
)

DEFAULT_PRICE_SERIES_LABEL = "Gold Price"

DEFAULT_ROLLING_HIGH_LABEL = (
    "{window}-Month Rolling High"
)

DEFAULT_ROLLING_LOW_LABEL = (
    "{window}-Month Rolling Low"
)

DEFAULT_ROLLING_HIGH_LOW_Y_AXIS_LABEL = (
    "Gold Price"
)

DEFAULT_RANGE_FILL_ALPHA = 0.10


# ------------------------------------------------------------------
# Forecast chart settings
# ------------------------------------------------------------------

DEFAULT_FORECAST_TITLE = (
    "Historical Gold Price with Growth Forecast"
)

DEFAULT_HISTORICAL_LABEL = (
    "Historical Gold Price"
)

DEFAULT_FORECAST_LABEL = (
    "Projected Gold Price"
)

DEFAULT_FORECAST_Y_AXIS_LABEL = (
    "Gold Price (USD per Troy Ounce)"
)

DEFAULT_FORECAST_LINE_STYLE = "--"

DEFAULT_FORECAST_LINE_WIDTH = 2.0

DEFAULT_FORECAST_MARKER_SIZE = 6.0

DEFAULT_FORECAST_BOUNDARY_LABEL = (
    "Forecast Begins"
)

DEFAULT_FORECAST_MONTHS_PER_YEAR = 12

DEFAULT_FORECAST_HISTORY_YEARS = 10

DEFAULT_FORECAST_PERIOD_YEARS = 5

# ------------------------------------------------------------------
# Shared Validation Helpers
# ------------------------------------------------------------------

def validate_figure_size(
    figure_size: tuple[float, float],
) -> None:
    """
    Validate a Matplotlib figure size.

    Args:
        figure_size:
            Tuple containing (width, height).

    Raises:
        TypeError:
            If figure_size is not a tuple of two numbers.

        ValueError:
            If either dimension is not positive.
    """

    if not isinstance(figure_size, tuple):
        raise TypeError(
            "figure_size must be a tuple."
        )

    if len(figure_size) != 2:
        raise ValueError(
            "figure_size must contain two values."
        )

    width, height = figure_size

    if not isinstance(width, (int, float)):
        raise TypeError(
            "Figure width must be numeric."
        )

    if not isinstance(height, (int, float)):
        raise TypeError(
            "Figure height must be numeric."
        )

    if width <= 0 or height <= 0:
        raise ValueError(
            "Figure dimensions must be greater than zero."
        )


def create_figure(
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Create a standard Matplotlib figure and axes.
    """

    validate_figure_size(figure_size)

    figure, axes = plt.subplots(
        figsize=figure_size
    )

    return figure, axes


def apply_standard_formatting(
    axes: Axes,
    *,
    title: str,
    x_label: str = DEFAULT_X_AXIS_LABEL,
    y_label: str = DEFAULT_Y_AXIS_LABEL,
    legend: bool = True,
) -> None:
    """
    Apply consistent formatting to every chart.
    """

    axes.set_title(
        title,
        pad=DEFAULT_TITLE_PADDING,
    )

    axes.set_xlabel(
        x_label,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.set_ylabel(
        y_label,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.grid(
        alpha=DEFAULT_GRID_ALPHA,
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    if legend:
        axes.legend()

    plt.tight_layout()


def format_currency_axis(
    axes: Axes,
) -> None:
    """
    Format the Y-axis as US dollars.
    """

    formatter = FuncFormatter(
        lambda value, _: f"${value:,.0f}"
    )

    axes.yaxis.set_major_formatter(
        formatter
    )


def format_percentage_axis(
    axes: Axes,
    *,
    axis: str = "y",
) -> None:
    """
    Format an axis as percentages.

    Args:
        axes:
            Matplotlib Axes object.

        axis:
            Either "x" or "y".
    """

    if axis not in {"x", "y"}:
        raise ValueError(
            "axis must be 'x' or 'y'."
        )

    formatter = FuncFormatter(
        lambda value, _: f"{value:.1%}"
    )

    if axis == "x":
        axes.xaxis.set_major_formatter(
            formatter
        )
    else:
        axes.yaxis.set_major_formatter(
            formatter
        )


def save_chart(
    figure: Figure,
    output_path: str | Path,
) -> None:
    """
    Save a chart to disk.
    """

    figure.savefig(
        output_path,
        dpi=DEFAULT_DPI,
        bbox_inches="tight",
    )

# ------------------------------------------------------------------
# Historical Price Charts
# ------------------------------------------------------------------

def plot_historical_price(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = DEFAULT_CHART_TITLE,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot the complete historical gold-price series.

    Args:
        data:
            DataFrame containing a Date column and the requested
            gold-price column.

        column:
            Name of the price column to plot.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        prices.index,
        prices.values,
        label=DEFAULT_PRICE_LABEL,
        linewidth=DEFAULT_LINE_WIDTH,
    )

    apply_standard_formatting(
        axes,
        title=title,
        y_label=DEFAULT_Y_AXIS_LABEL,
    )

    format_currency_axis(axes)

    return figure, axes


def plot_recent_price(
    data: pd.DataFrame,
    years: int = DEFAULT_RECENT_YEARS,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = DEFAULT_RECENT_PRICE_TITLE,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot gold prices from the most recent requested number of years.

    Args:
        data:
            DataFrame containing a Date column and the requested
            gold-price column.

        years:
            Number of recent calendar years to include.

        column:
            Name of the price column to plot.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If years is not an integer.

        ValueError:
            If years is not greater than zero or the requested
            date range contains no observations.
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

    latest_date = prices.index.max()

    start_date = latest_date - pd.DateOffset(
        years=years
    )

    recent_prices = prices.loc[
        prices.index >= start_date
    ]

    if recent_prices.empty:
        raise ValueError(
            "No price observations were found in the "
            "requested recent period."
        )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        recent_prices.index,
        recent_prices.values,
        label=f"Most Recent {years} Years",
        linewidth=DEFAULT_LINE_WIDTH,
    )

    apply_standard_formatting(
        axes,
        title=f"{title}: Last {years} Years",
        y_label=DEFAULT_Y_AXIS_LABEL,
    )

    format_currency_axis(axes)

    return figure, axes

# ------------------------------------------------------------------
# Inflation-Adjusted Charts
# ------------------------------------------------------------------

def plot_inflation_adjusted_price(
    data: pd.DataFrame,
    *,
    base_date: str | pd.Timestamp |None = None,
    title: str = DEFAULT_INFLATION_ADJUSTED_TITLE,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot inflation-adjusted gold prices.
    """

    real_prices = calculate_real_price(
        data=data,
        base_date=base_date,
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        real_prices.index,
        real_prices.values,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_REAL_LABEL,
    )

    apply_standard_formatting(
        axes,
        title=title,
        y_label=DEFAULT_INFLATION_ADJUSTED_Y_AXIS_LABEL,
    )

    format_currency_axis(axes)

    return figure, axes


def plot_nominal_vs_real_price(
    data: pd.DataFrame,
    *,
    base_date: str | pd.Timestamp | None = None,
    title: str = DEFAULT_NOMINAL_VS_REAL_TITLE,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Compare nominal and inflation-adjusted gold prices.
    """

    nominal_prices = prepare_price_series(data)

    real_prices = calculate_real_price(
        data=data,
        base_date=base_date,
    )

    comparison = pd.concat(
        [
            nominal_prices.rename(
                DEFAULT_NOMINAL_LABEL
            ),
            real_prices.rename(
                DEFAULT_REAL_LABEL
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        comparison.index,
        comparison[
            DEFAULT_NOMINAL_LABEL
        ],
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_NOMINAL_LABEL,
    )

    axes.plot(
        comparison.index,
        comparison[
            DEFAULT_REAL_LABEL
        ],
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_REAL_LABEL,
    )

    apply_standard_formatting(
        axes,
        title=title,
        y_label=DEFAULT_COMPARISON_Y_AXIS_LABEL,
    )

    format_currency_axis(axes)

    return figure, axes

# ------------------------------------------------------------------
# Return Charts
# ------------------------------------------------------------------

def plot_monthly_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = "Monthly Gold Returns",
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot month-over-month returns as a time series.

    Returns are represented as decimal values by the analytics
    layer and displayed as percentages on the chart.

    Args:
        data:
            Source DataFrame containing dates and price values.

        column:
            Numeric price column used to calculate returns.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.
    """

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if monthly_returns.empty:
        raise ValueError(
            "No monthly returns are available to plot."
        )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        monthly_returns.index,
        monthly_returns.values,
        linewidth=DEFAULT_LINE_WIDTH,
        label="Monthly Return",
    )

    axes.axhline(
        y=0,
        linewidth=1,
        linestyle="--",
    )

    apply_standard_formatting(
        axes,
        title=title,
        y_label="Monthly Return",
    )

    format_percentage_axis(axes)

    return figure, axes


def plot_cumulative_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = "Cumulative Gold Returns",
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot cumulative growth relative to the first observation.

    A cumulative return of 1.00 represents growth of 100%
    relative to the starting value.

    Args:
        data:
            Source DataFrame containing dates and price values.

        column:
            Numeric price column used to calculate returns.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.
    """

    cumulative_returns = calculate_cumulative_returns(
        data=data,
        column=column,
    )

    if cumulative_returns.empty:
        raise ValueError(
            "No cumulative returns are available to plot."
        )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        cumulative_returns.index,
        cumulative_returns.values,
        linewidth=DEFAULT_LINE_WIDTH,
        label="Cumulative Return",
    )

    axes.axhline(
        y=0,
        linewidth=1,
        linestyle="--",
    )

    apply_standard_formatting(
        axes,
        title=title,
        y_label="Cumulative Return",
    )

    format_percentage_axis(axes)

    return figure, axes


def plot_annual_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = "Annual Gold Returns",
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot calendar-year gold returns as a bar chart.

    Each annual return compares the final available price in
    one year with the final available price in the previous
    year.

    Args:
        data:
            Source DataFrame containing dates and price values.

        column:
            Numeric price column used to calculate returns.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.
    """

    annual_returns = calculate_annual_returns(
        data=data,
        column=column,
    )

    if annual_returns.empty:
        raise ValueError(
            "No annual returns are available to plot."
        )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.bar(
        annual_returns.index,
        annual_returns.values,
        label="Annual Return",
    )

    axes.axhline(
        y=0,
        linewidth=1,
    )

    apply_standard_formatting(
        axes,
        title=title,
        x_label="Year",
        y_label="Annual Return",
    )

    format_percentage_axis(axes)

    return figure, axes


def plot_return_distribution(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    bins: int = 40,
    title: str = "Distribution of Monthly Gold Returns",
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot the frequency distribution of monthly returns.

    Args:
        data:
            Source DataFrame containing dates and price values.

        column:
            Numeric price column used to calculate returns.

        bins:
            Number of histogram intervals.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If bins is not an integer.

        ValueError:
            If bins is below one or no monthly returns exist.
    """

    if isinstance(bins, bool) or not isinstance(
        bins,
        int,
    ):
        raise TypeError(
            "bins must be an integer."
        )

    if bins < 1:
        raise ValueError(
            "bins must be at least 1."
        )

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if monthly_returns.empty:
        raise ValueError(
            "No monthly returns are available to plot."
        )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.hist(
        monthly_returns.values,
        bins=bins,
        label="Monthly Returns",
    )

    axes.axvline(
        x=0,
        linewidth=1,
        linestyle="--",
    )

    apply_standard_formatting(
        axes,
        title=title,
        x_label="Monthly Return",
        y_label="Frequency",
    )

    format_percentage_axis(
        axes,
        axis="x",
    )

    return figure, axes

# ------------------------------------------------------------------
# Rolling Analysis Charts
# ------------------------------------------------------------------

def plot_rolling_average(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    *,
    title: str = DEFAULT_ROLLING_AVERAGE_TITLE,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot gold prices alongside a rolling arithmetic average.

    Args:
        data:
            Source DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly observations in the rolling window.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    rolling_average = calculate_rolling_average(
        data=data,
        column=column,
        window=window,
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        prices.index,
        prices.values,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_PRICE_LABEL,
    )

    axes.plot(
        rolling_average.index,
        rolling_average.values,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_AVERAGE_LABEL.format(
            window=window
        ),
    )

    apply_standard_formatting(
        axes,
        title=f"{title}: {window}-Month Window",
        y_label=DEFAULT_ROLLING_Y_AXIS_LABEL,
    )

    format_currency_axis(axes)

    return figure, axes


def plot_rolling_volatility(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    *,
    annualize: bool = True,
    title: str = DEFAULT_ROLLING_VOLATILITY_TITLE,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot rolling volatility calculated from monthly returns.

    Args:
        data:
            Source DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly returns in the rolling window.

        annualize:
            Whether to annualize the volatility calculation.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If annualize is not a Boolean value.
    """

    if not isinstance(annualize, bool):
        raise TypeError(
            "annualize must be a Boolean value."
        )

    rolling_volatility = calculate_rolling_volatility(
        data=data,
        column=column,
        window=window,
        annualize=annualize,
    )

    valid_volatility = rolling_volatility.dropna()

    if valid_volatility.empty:
        raise ValueError(
            "No rolling volatility values are available to plot."
        )

    if annualize:
        series_label = (
            DEFAULT_ROLLING_VOLATILITY_LABEL.format(
                window=window
            )
        )

        y_axis_label = (
            DEFAULT_VOLATILITY_Y_AXIS_LABEL
        )

    else:
        series_label = (
            DEFAULT_NON_ANNUALIZED_VOLATILITY_LABEL.format(
                window=window
            )
        )

        y_axis_label = (
            DEFAULT_NON_ANNUALIZED_VOLATILITY_Y_AXIS_LABEL
        )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        valid_volatility.index,
        valid_volatility.values,
        linewidth=DEFAULT_LINE_WIDTH,
        label=series_label,
    )

    apply_standard_formatting(
        axes,
        title=f"{title}: {window}-Month Window",
        y_label=y_axis_label,
    )

    format_percentage_axis(axes)

    return figure, axes


def plot_rolling_return(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    *,
    title: str = DEFAULT_ROLLING_RETURN_TITLE,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot percentage returns measured across rolling periods.

    A 12-month window compares each price with the price
    12 monthly observations earlier.

    Args:
        data:
            Source DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of periods across which return is measured.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.
    """

    rolling_return = calculate_rolling_return(
        data=data,
        column=column,
        window=window,
    )

    valid_returns = rolling_return.dropna()

    if valid_returns.empty:
        raise ValueError(
            "No rolling return values are available to plot."
        )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        valid_returns.index,
        valid_returns.values,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_RETURN_LABEL.format(
            window=window
        ),
    )

    axes.axhline(
        y=0,
        linewidth=DEFAULT_REFERENCE_LINE_WIDTH,
        linestyle="--",
    )

    apply_standard_formatting(
        axes,
        title=f"{title}: {window}-Month Window",
        y_label=DEFAULT_ROLLING_RETURN_Y_AXIS_LABEL,
    )

    format_percentage_axis(axes)

    return figure, axes


def plot_rolling_drawdown(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    *,
    title: str = DEFAULT_ROLLING_DRAWDOWN_TITLE,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot each price's decline from its rolling-window high.

    A value of -0.15 means the current price is 15% below
    the highest price observed during the rolling window.

    Args:
        data:
            Source DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly observations in the rolling window.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.
    """

    rolling_drawdown = calculate_rolling_drawdown(
        data=data,
        column=column,
        window=window,
    )

    valid_drawdown = rolling_drawdown.dropna()

    if valid_drawdown.empty:
        raise ValueError(
            "No rolling drawdown values are available to plot."
        )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        valid_drawdown.index,
        valid_drawdown.values,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_DRAWDOWN_LABEL.format(
            window=window
        ),
    )

    axes.fill_between(
        valid_drawdown.index,
        valid_drawdown.values,
        0,
        alpha=DEFAULT_PERCENTAGE_FILL_ALPHA,
    )

    axes.axhline(
        y=0,
        linewidth=DEFAULT_REFERENCE_LINE_WIDTH,
    )

    apply_standard_formatting(
        axes,
        title=f"{title}: {window}-Month Window",
        y_label=DEFAULT_ROLLING_DRAWDOWN_Y_AXIS_LABEL,
    )

    format_percentage_axis(axes)

    return figure, axes


def plot_rolling_high_low(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    *,
    title: str = DEFAULT_ROLLING_HIGH_LOW_TITLE,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot gold prices with their rolling highs and lows.

    The area between the rolling high and rolling low shows
    the price range observed within each rolling window.

    Args:
        data:
            Source DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly observations in the rolling window.

        title:
            Title displayed above the chart.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the Matplotlib Figure and Axes.
    """

    prices = prepare_price_series(
        data=data,
        column=column,
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

    rolling_range = pd.concat(
        [
            rolling_high.rename("Rolling High"),
            rolling_low.rename("Rolling Low"),
        ],
        axis=1,
    ).dropna()

    if rolling_range.empty:
        raise ValueError(
            "No rolling high and low values are available "
            "to plot."
        )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        prices.index,
        prices.values,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_PRICE_SERIES_LABEL,
    )

    axes.plot(
        rolling_range.index,
        rolling_range["Rolling High"],
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_HIGH_LABEL.format(
            window=window
        ),
    )

    axes.plot(
        rolling_range.index,
        rolling_range["Rolling Low"],
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_LOW_LABEL.format(
            window=window
        ),
    )

    axes.fill_between(
        rolling_range.index,
        rolling_range["Rolling Low"],
        rolling_range["Rolling High"],
        alpha=DEFAULT_RANGE_FILL_ALPHA,
    )

    apply_standard_formatting(
        axes,
        title=f"{title}: {window}-Month Window",
        y_label=DEFAULT_ROLLING_HIGH_LOW_Y_AXIS_LABEL,
    )

    format_currency_axis(axes)

    return figure, axes