"""
Reusable chart functions for the Gold Value Predictor.

Each plotting function returns its Matplotlib Figure and Axes
objects instead of displaying the chart automatically. This
allows charts to be tested, customized, displayed, or saved by
other parts of the application.
"""

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

from analytics.forecasting import (
    DEFAULT_EXPECTED_GROWTH_RATE,
    DEFAULT_FORECAST_SCENARIOS,
    DEFAULT_FORECAST_YEARS,
    generate_forecast_scenarios,
    generate_forecast_series,
)

from visualizations.styling import (
    ANNUAL_RETURN_COLOR,
    CUMULATIVE_RETURN_COLOR,
    DEFAULT_LINE_WIDTH,
    DRAWDOWN_COLOR,
    HISTORICAL_PRICE_COLOR,
    NEGATIVE_RETURN_COLOR,
    NEUTRAL_RETURN_COLOR,
    NOMINAL_PRICE_COLOR,
    POSITIVE_RETURN_COLOR,
    REAL_PRICE_COLOR,
    RECENT_PRICE_COLOR,
    REFERENCE_LINE_COLOR,
    ROLLING_AVERAGE_COLOR,
    ROLLING_HIGH_COLOR,
    ROLLING_LOW_COLOR,
    ROLLING_RANGE_COLOR,
    ROLLING_RETURN_COLOR,
    VOLATILITY_COLOR,
    apply_standard_formatting,
    FORECAST_BOUNDARY_COLOR,
    FORECAST_COLOR,
    FORECAST_CONSERVATIVE_COLOR,
    FORECAST_BASE_COLOR,
    FORECAST_OPTIMISTIC_COLOR,
    SCENARIO_COLORS,
    SECONDARY_TEXT_COLOR,
    annotate_high_value,
    annotate_latest_value,
    annotate_low_value,
    GOLD_HIGHLIGHT_COLOR,
    annotate_vertical_event,
    add_figure_source_note,
    add_watermark,
)

# ------------------------------------------------------------------
# General chart settings
# ------------------------------------------------------------------

DEFAULT_CHART_TITLE = "Historical Gold Prices"

DEFAULT_X_AXIS_LABEL = "Date"

DEFAULT_Y_AXIS_LABEL = "Gold Price (USD)"

DEFAULT_FIGURE_SIZE = (12.0, 6.0)


# ------------------------------------------------------------------
# Recent-price chart settings
# ------------------------------------------------------------------

DEFAULT_RECENT_YEARS = 5

DEFAULT_RECENT_PRICE_TITLE = "Recent Gold Prices"


# ------------------------------------------------------------------
# Inflation-adjusted chart settings
# ------------------------------------------------------------------

DEFAULT_INFLATION_ADJUSTED_TITLE = (
    "Inflation-Adjusted Gold Prices"
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

DEFAULT_RECENT_PRICE_SUBTITLE = (
    "Monthly gold prices from the most recent "
    "{years} years"
)

DEFAULT_INFLATION_ADJUSTED_SUBTITLE = (
    "Historical gold prices expressed in constant "
    "purchasing-power dollars"
)

DEFAULT_NOMINAL_VS_REAL_SUBTITLE = (
    "Comparison of market prices and "
    "inflation-adjusted purchasing power"
)

# ------------------------------------------------------------------
# Return chart settings
# ------------------------------------------------------------------

DEFAULT_MONTHLY_RETURNS_TITLE = (
    "Monthly Gold Price Returns"
)


DEFAULT_RETURNS_Y_AXIS_LABEL = (
    "Monthly Return"
)

DEFAULT_REFERENCE_LINE_WIDTH = 1.25

DEFAULT_PERCENTAGE_FILL_ALPHA = 0.20

DEFAULT_MONTHLY_RETURNS_SUBTITLE = (
    "Month-over-month percentage changes in gold prices"
)

DEFAULT_CUMULATIVE_RETURNS_SUBTITLE = (
    "Total percentage growth relative to the first observation"
)

DEFAULT_ANNUAL_RETURNS_SUBTITLE = (
    "Calendar-year performance based on year-end gold prices"
)

DEFAULT_RETURN_DISTRIBUTION_SUBTITLE = (
    "Frequency distribution of monthly gold-price returns"
)


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

DEFAULT_ROLLING_AVERAGE_SUBTITLE = (
    "Gold prices compared with their "
    "{window}-month moving average"
)

DEFAULT_ROLLING_VOLATILITY_SUBTITLE = (
    "{window}-month {volatility_type} volatility "
    "calculated from monthly returns"
)

DEFAULT_ROLLING_RETURN_SUBTITLE = (
    "Percentage change across each "
    "{window}-month rolling period"
)

DEFAULT_ROLLING_DRAWDOWN_SUBTITLE = (
    "Decline from the highest gold price observed "
    "within each {window}-month window"
)

DEFAULT_ROLLING_HIGH_LOW_SUBTITLE = (
    "Gold prices compared with the highest and lowest "
    "values in each {window}-month window"
)


# ------------------------------------------------------------------
# Forecast chart defaults
# ------------------------------------------------------------------

DEFAULT_FORECAST_TITLE = (
    "Historical Gold Price with Forecast"
)

DEFAULT_FORECAST_SCENARIO_TITLE = (
    "Gold Price Forecast Scenarios"
)

DEFAULT_FORECAST_Y_AXIS_LABEL = (
    "Gold Price (USD)"
)

DEFAULT_FORECAST_HISTORY_LABEL = (
    "Historical Price"
)

DEFAULT_FORECAST_LABEL = (
    "Projected Price"
)

DEFAULT_FORECAST_BOUNDARY_LABEL = (
    "Forecast Begins"
)

DEFAULT_FORECAST_HISTORY_YEARS = 10

DEFAULT_FORECAST_PERIOD_YEARS = (
    DEFAULT_FORECAST_YEARS
)

DEFAULT_FORECAST_LINE_STYLE = "--"

DEFAULT_FORECAST_BOUNDARY_STYLE = ":"

DEFAULT_FORECAST_BOUNDARY_WIDTH = 1.5

DEFAULT_FORECAST_MARKER_SIZE = 5.0

DEFAULT_FORECAST_SUBTITLE = (
    "Hypothetical {forecast_years}-year projection using "
    "a constant {growth_rate:.1%} annual growth assumption"
)

DEFAULT_FORECAST_SCENARIO_SUBTITLE = (
    "Comparison of hypothetical {forecast_years}-year "
    "growth scenarios"
)

DEFAULT_FORECAST_DISCLAIMER = (
    "Scenario projections are illustrative and are not "
    "predictions or investment advice."
)

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

# ------------------------------------------------------------------
# Historical Price Charts
# ------------------------------------------------------------------

DEFAULT_HISTORICAL_PRICE_SUBTITLE = (
    "Monthly historical gold prices in U.S. dollars"
)

def plot_historical_price(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = DEFAULT_CHART_TITLE,
    subtitle: str | None = DEFAULT_HISTORICAL_PRICE_SUBTITLE,
    show_latest_value: bool = True,
    source_note: str | None = None,
    watermark: str | None = None,
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

        subtitle:
            Optional descriptive text displayed beneath the title.
            Pass None to omit the subtitle.

        show_latest_value:
            Whether to annotate the latest valid gold price.

        source_note:
            Optional source or attribution text displayed along
            the bottom of the Figure.

        watermark:
            Optional subtle watermark displayed inside the Axes.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If show_latest_value is not a Boolean value.
    """

    if not isinstance(
        show_latest_value,
        bool,
    ):
        raise TypeError(
            "show_latest_value must be a Boolean value."
        )

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
        color=HISTORICAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=title,
        subtitle=subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_currency_axis(
        axes
    )

    if show_latest_value:
        annotate_latest_value(
            axes,
            prices,
            prefix="Latest",
            value_format="currency",
            marker_color=GOLD_HIGHLIGHT_COLOR,
            offset=(
                -90,
                14,
            ),
        )

    if watermark is not None:
        add_watermark(
            axes,
            watermark,
        )

    if source_note is not None:
        add_figure_source_note(
            figure,
            source_note,
        )

    figure.tight_layout(
        pad=1.5,
        rect=(
            0.0,
            0.04 if source_note is not None else 0.0,
            1.0,
            1.0,
        ),
    )

    return figure, axes


def plot_recent_price(
    data: pd.DataFrame,
    years: int = DEFAULT_RECENT_YEARS,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = DEFAULT_RECENT_PRICE_TITLE,
    subtitle: str | None = None,
    show_latest_value: bool = True,
    show_high_low: bool = True,
    source_note: str | None = None,
    watermark: str | None = None,
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
            Primary chart title.

        subtitle:
            Optional subtitle. When omitted, a subtitle describing
            the selected recent period is generated automatically.

        show_latest_value:
            Whether to annotate the latest valid price.

        show_high_low:
            Whether to annotate the period high and low.

        source_note:
            Optional source or attribution text displayed along
            the bottom of the Figure.

        watermark:
            Optional subtle watermark displayed inside the Axes.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If years is not an integer or either annotation option
            is not a Boolean value.

        ValueError:
            If years is not greater than zero or the requested
            period contains no observations.
    """

    if (
        isinstance(years, bool)
        or not isinstance(years, int)
    ):
        raise TypeError(
            "years must be an integer."
        )

    if years <= 0:
        raise ValueError(
            "years must be greater than zero."
        )

    if not isinstance(
        show_latest_value,
        bool,
    ):
        raise TypeError(
            "show_latest_value must be a Boolean value."
        )

    if not isinstance(
        show_high_low,
        bool,
    ):
        raise TypeError(
            "show_high_low must be a Boolean value."
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

    resolved_subtitle = (
        subtitle
        if subtitle is not None
        else DEFAULT_RECENT_PRICE_SUBTITLE.format(
            years=years
        )
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        recent_prices.index,
        recent_prices.values,
        label=f"Most Recent {years} Years",
        color=RECENT_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=f"{title}: Last {years} Years",
        subtitle=resolved_subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_currency_axis(
        axes
    )

    latest_observation_date = (
        recent_prices.index[-1]
    )

    high_date = recent_prices.idxmax()
    low_date = recent_prices.idxmin()

    if show_latest_value:
        annotate_latest_value(
            axes,
            recent_prices,
            prefix="Latest",
            value_format="currency",
            marker_color=RECENT_PRICE_COLOR,
            offset=(
                -90,
                14,
            ),
        )

    if show_high_low:
        if high_date != latest_observation_date:
            annotate_high_value(
                axes,
                recent_prices,
                prefix=f"{years}-Year High",
                value_format="currency",
                marker_color=GOLD_HIGHLIGHT_COLOR,
                offset=(
                    12,
                    16,
                ),
            )

        if low_date != latest_observation_date:
            annotate_low_value(
                axes,
                recent_prices,
                prefix=f"{years}-Year Low",
                value_format="currency",
                marker_color=HISTORICAL_PRICE_COLOR,
                offset=(
                    12,
                    -28,
                ),
            )

    if watermark is not None:
        add_watermark(
            axes,
            watermark,
        )

    if source_note is not None:
        add_figure_source_note(
            figure,
            source_note,
        )

    figure.tight_layout(
        pad=1.5,
        rect=(
            0.0,
            0.04 if source_note is not None else 0.0,
            1.0,
            1.0,
        ),
    )

    return figure, axes

# ------------------------------------------------------------------
# Inflation-Adjusted Charts
# ------------------------------------------------------------------

def plot_inflation_adjusted_price(
    data: pd.DataFrame,
    *,
    base_date: str | pd.Timestamp | None = None,
    title: str = DEFAULT_INFLATION_ADJUSTED_TITLE,
    subtitle: str | None = (
        DEFAULT_INFLATION_ADJUSTED_SUBTITLE
    ),
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot inflation-adjusted gold prices.

    Args:
        data:
            DataFrame containing Date, gold-price, and CPI data.

        base_date:
            Optional purchasing-power base period. When omitted,
            the latest available CPI observation is used.

        title:
            Primary chart title.

        subtitle:
            Optional chart subtitle. Pass None to omit it.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Matplotlib Figure and Axes.
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
        color=REAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_REAL_LABEL,
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=title,
        subtitle=subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=(
            DEFAULT_INFLATION_ADJUSTED_Y_AXIS_LABEL
        ),
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=True,
    )

    format_currency_axis(
        axes
    )

    return figure, axes


def plot_nominal_vs_real_price(
    data: pd.DataFrame,
    *,
    base_date: str | pd.Timestamp | None = None,
    title: str = DEFAULT_NOMINAL_VS_REAL_TITLE,
    subtitle: str | None = (
        DEFAULT_NOMINAL_VS_REAL_SUBTITLE
    ),
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Compare nominal and inflation-adjusted gold prices.

    Args:
        data:
            DataFrame containing Date, gold-price, and CPI data.

        base_date:
            Optional purchasing-power base period. When omitted,
            the latest available CPI observation is used.

        title:
            Primary chart title.

        subtitle:
            Optional chart subtitle. Pass None to omit it.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Matplotlib Figure and Axes.
    """

    nominal_prices = prepare_price_series(
        data
    )

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
        color=NOMINAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_NOMINAL_LABEL,
    )

    axes.plot(
        comparison.index,
        comparison[
            DEFAULT_REAL_LABEL
        ],
        color=REAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_REAL_LABEL,
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=title,
        subtitle=subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_COMPARISON_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=True,
    )

    format_currency_axis(
        axes
    )

    return figure, axes

# ------------------------------------------------------------------
# Return Charts
# ------------------------------------------------------------------

def plot_monthly_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = DEFAULT_MONTHLY_RETURNS_TITLE,
    subtitle: str | None = (
        DEFAULT_MONTHLY_RETURNS_SUBTITLE
    ),
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot month-over-month gold-price returns.

    Positive and negative returns are displayed using separate
    semantic colors, making changes in direction easier to read.

    Args:
        data:
            Source DataFrame containing dates and price values.

        column:
            Numeric price column used to calculate returns.

        title:
            Primary chart title.

        subtitle:
            Optional descriptive subtitle. Pass None to omit it.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Matplotlib Figure and Axes.
    """

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=column,
    )

    if monthly_returns.empty:
        raise ValueError(
            "No monthly returns are available to plot."
        )

    positive_returns = monthly_returns.where(
        monthly_returns >= 0
    )

    negative_returns = monthly_returns.where(
        monthly_returns < 0
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        positive_returns.index,
        positive_returns.values,
        color=POSITIVE_RETURN_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label="Positive Monthly Return",
    )

    axes.plot(
        negative_returns.index,
        negative_returns.values,
        color=NEGATIVE_RETURN_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label="Negative Monthly Return",
    )

    axes.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=DEFAULT_REFERENCE_LINE_WIDTH,
        linestyle="--",
        label="Zero Return",
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=title,
        subtitle=subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_RETURNS_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=True,
    )

    format_percentage_axis(
        axes
    )

    return figure, axes

def plot_cumulative_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = "Cumulative Gold Returns",
    subtitle: str | None = (
        DEFAULT_CUMULATIVE_RETURNS_SUBTITLE
    ),
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot cumulative gold-price returns.

    A cumulative return of 1.0 represents growth of 100%
    relative to the first usable observation.

    Args:
        data:
            Source DataFrame containing dates and price values.

        column:
            Numeric price column used to calculate returns.

        title:
            Primary chart title.

        subtitle:
            Optional descriptive subtitle. Pass None to omit it.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Matplotlib Figure and Axes.
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
        color=CUMULATIVE_RETURN_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label="Cumulative Return",
    )

    axes.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=DEFAULT_REFERENCE_LINE_WIDTH,
        linestyle="--",
        label="Starting Level",
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=title,
        subtitle=subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label="Cumulative Return",
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=True,
    )

    format_percentage_axis(
        axes
    )

    return figure, axes


def plot_annual_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    title: str = "Annual Gold Returns",
    subtitle: str | None = (
        DEFAULT_ANNUAL_RETURNS_SUBTITLE
    ),
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot calendar-year gold returns as a bar chart.

    Positive years use the shared positive-return color and
    negative years use the shared negative-return color.

    Args:
        data:
            Source DataFrame containing dates and price values.

        column:
            Numeric price column used to calculate returns.

        title:
            Primary chart title.

        subtitle:
            Optional descriptive subtitle. Pass None to omit it.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Matplotlib Figure and Axes.
    """

    annual_returns = calculate_annual_returns(
        data=data,
        column=column,
    )

    if annual_returns.empty:
        raise ValueError(
            "No annual returns are available to plot."
        )

    bar_colors = [
        (
            POSITIVE_RETURN_COLOR
            if value >= 0
            else NEGATIVE_RETURN_COLOR
        )
        for value in annual_returns.values
    ]

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.bar(
        annual_returns.index,
        annual_returns.values,
        color=bar_colors,
        label="Annual Return",
    )

    axes.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=DEFAULT_REFERENCE_LINE_WIDTH,
        linestyle="--",
    )

    apply_standard_formatting(
        figure,
        axes,
        title=title,
        subtitle=subtitle,
        x_label="Year",
        y_label="Annual Return",
        show_grid=True,
        show_legend=False,
        apply_tight_layout=True,
    )

    format_percentage_axis(
        axes
    )

    return figure, axes


def plot_return_distribution(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    bins: int = 40,
    title: str = "Distribution of Monthly Gold Returns",
    subtitle: str | None = (
        DEFAULT_RETURN_DISTRIBUTION_SUBTITLE
    ),
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
            Primary chart title.

        subtitle:
            Optional descriptive subtitle. Pass None to omit it.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If bins is not an integer.

        ValueError:
            If bins is less than one or no returns exist.
    """

    if (
        isinstance(bins, bool)
        or not isinstance(bins, int)
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
        color=NEUTRAL_RETURN_COLOR,
        alpha=DEFAULT_PERCENTAGE_FILL_ALPHA,
        label="Monthly Returns",
    )

    axes.axvline(
        x=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=DEFAULT_REFERENCE_LINE_WIDTH,
        linestyle="--",
        label="Zero Return",
    )

    apply_standard_formatting(
        figure,
        axes,
        title=title,
        subtitle=subtitle,
        x_label="Monthly Return",
        y_label="Frequency",
        show_grid=True,
        show_legend=True,
        legend_location="upper right",
        apply_tight_layout=True,
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
    subtitle: str | None = None,
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
            Primary chart title.

        subtitle:
            Optional subtitle. When omitted, a description of
            the selected rolling window is generated.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Figure and Axes.
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

    resolved_subtitle = (
        subtitle
        if subtitle is not None
        else DEFAULT_ROLLING_AVERAGE_SUBTITLE.format(
            window=window
        )
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        prices.index,
        prices.values,
        color=HISTORICAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_PRICE_LABEL,
    )

    axes.plot(
        rolling_average.index,
        rolling_average.values,
        color=ROLLING_AVERAGE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_AVERAGE_LABEL.format(
            window=window
        ),
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=f"{title}: {window}-Month Window",
        subtitle=resolved_subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_ROLLING_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=True,
    )

    format_currency_axis(
        axes
    )

    return figure, axes


def plot_rolling_volatility(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    *,
    annualize: bool = True,
    title: str = DEFAULT_ROLLING_VOLATILITY_TITLE,
    subtitle: str | None = None,
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
            Primary chart title.

        subtitle:
            Optional subtitle. When omitted, one is generated
            from the window and annualization setting.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Figure and Axes.

    Raises:
        TypeError:
            If annualize is not a Boolean.
    """

    if not isinstance(
        annualize,
        bool,
    ):
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

        volatility_type = "annualized"

    else:
        series_label = (
            DEFAULT_NON_ANNUALIZED_VOLATILITY_LABEL.format(
                window=window
            )
        )

        y_axis_label = (
            DEFAULT_NON_ANNUALIZED_VOLATILITY_Y_AXIS_LABEL
        )

        volatility_type = "monthly"

    resolved_subtitle = (
        subtitle
        if subtitle is not None
        else DEFAULT_ROLLING_VOLATILITY_SUBTITLE.format(
            window=window,
            volatility_type=volatility_type,
        )
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        valid_volatility.index,
        valid_volatility.values,
        color=VOLATILITY_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=series_label,
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=f"{title}: {window}-Month Window",
        subtitle=resolved_subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=y_axis_label,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=True,
    )

    format_percentage_axis(
        axes
    )

    return figure, axes


def plot_rolling_return(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    *,
    title: str = DEFAULT_ROLLING_RETURN_TITLE,
    subtitle: str | None = None,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot percentage returns measured across rolling periods.

    Args:
        data:
            Source DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of periods across which return is measured.

        title:
            Primary chart title.

        subtitle:
            Optional subtitle. When omitted, one is generated
            from the selected rolling window.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Figure and Axes.
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

    resolved_subtitle = (
        subtitle
        if subtitle is not None
        else DEFAULT_ROLLING_RETURN_SUBTITLE.format(
            window=window
        )
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        valid_returns.index,
        valid_returns.values,
        color=ROLLING_RETURN_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_RETURN_LABEL.format(
            window=window
        ),
    )

    axes.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=DEFAULT_REFERENCE_LINE_WIDTH,
        linestyle="--",
        label="Zero Return",
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=f"{title}: {window}-Month Window",
        subtitle=resolved_subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_ROLLING_RETURN_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=True,
    )

    format_percentage_axis(
        axes
    )

    return figure, axes


def plot_rolling_drawdown(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    *,
    title: str = DEFAULT_ROLLING_DRAWDOWN_TITLE,
    subtitle: str | None = None,
    show_deepest_drawdown: bool = True,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot each price's decline from its rolling-window high.

    Args:
        data:
            Source DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly observations in the rolling window.

        title:
            Primary chart title.

        subtitle:
            Optional subtitle. When omitted, one is generated
            from the selected rolling window.

        show_deepest_drawdown:
            Whether to annotate the most negative drawdown.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Figure and Axes.

    Raises:
        TypeError:
            If show_deepest_drawdown is not a Boolean value.
    """

    if not isinstance(
        show_deepest_drawdown,
        bool,
    ):
        raise TypeError(
            "show_deepest_drawdown must be a Boolean value."
        )

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

    resolved_subtitle = (
        subtitle
        if subtitle is not None
        else DEFAULT_ROLLING_DRAWDOWN_SUBTITLE.format(
            window=window
        )
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        valid_drawdown.index,
        valid_drawdown.values,
        color=DRAWDOWN_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_DRAWDOWN_LABEL.format(
            window=window
        ),
    )

    axes.fill_between(
        valid_drawdown.index,
        valid_drawdown.values,
        0,
        color=DRAWDOWN_COLOR,
        alpha=DEFAULT_PERCENTAGE_FILL_ALPHA,
    )

    axes.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=DEFAULT_REFERENCE_LINE_WIDTH,
        linestyle="--",
        label="Rolling High",
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=f"{title}: {window}-Month Window",
        subtitle=resolved_subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_ROLLING_DRAWDOWN_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="lower left",
        apply_tight_layout=True,
    )

    format_percentage_axis(
        axes
    )

    if show_deepest_drawdown:
        annotate_low_value(
            axes,
            valid_drawdown,
            prefix="Deepest Drawdown",
            value_format="percentage",
            marker_color=DRAWDOWN_COLOR,
            offset=(
                12,
                -30,
            ),
        )

    return figure, axes


def plot_rolling_high_low(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    *,
    title: str = DEFAULT_ROLLING_HIGH_LOW_TITLE,
    subtitle: str | None = None,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
) -> tuple[Figure, Axes]:
    """
    Plot gold prices with their rolling highs and lows.

    Args:
        data:
            Source DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly observations in the rolling window.

        title:
            Primary chart title.

        subtitle:
            Optional subtitle. When omitted, one is generated
            from the selected rolling window.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Figure and Axes.
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

    resolved_subtitle = (
        subtitle
        if subtitle is not None
        else DEFAULT_ROLLING_HIGH_LOW_SUBTITLE.format(
            window=window
        )
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        prices.index,
        prices.values,
        color=HISTORICAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_PRICE_SERIES_LABEL,
    )

    axes.plot(
        rolling_range.index,
        rolling_range["Rolling High"],
        color=ROLLING_HIGH_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_HIGH_LABEL.format(
            window=window
        ),
    )

    axes.plot(
        rolling_range.index,
        rolling_range["Rolling Low"],
        color=ROLLING_LOW_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_ROLLING_LOW_LABEL.format(
            window=window
        ),
    )

    axes.fill_between(
        rolling_range.index,
        rolling_range["Rolling Low"],
        rolling_range["Rolling High"],
        color=ROLLING_RANGE_COLOR,
        alpha=DEFAULT_RANGE_FILL_ALPHA,
        label=f"{window}-Month Price Range",
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=f"{title}: {window}-Month Window",
        subtitle=resolved_subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_ROLLING_HIGH_LOW_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=True,
    )

    format_currency_axis(
        axes
    )

    return figure, axes

# ------------------------------------------------------------------
# Forecast Charts
# ------------------------------------------------------------------

def plot_forecast(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    annual_growth_rate: float = (
        DEFAULT_EXPECTED_GROWTH_RATE
    ),
    forecast_years: int = (
        DEFAULT_FORECAST_PERIOD_YEARS
    ),
    history_years: int | None = (
        DEFAULT_FORECAST_HISTORY_YEARS
    ),
    title: str = DEFAULT_FORECAST_TITLE,
    subtitle: str | None = None,
    show_disclaimer: bool = True,
    show_projected_value: bool = True,
    source_note: str | None = None,
    watermark: str | None = None,
    figure_size: tuple[float, float] = (
        DEFAULT_FIGURE_SIZE
    ),
) -> tuple[Figure, Axes]:
    """
    Plot historical gold prices followed by one hypothetical
    compound-growth forecast.

    Args:
        data:
            Source DataFrame containing historical prices.

        column:
            Numeric price column used for the chart.

        annual_growth_rate:
            Constant annual forecast growth rate expressed
            as a decimal.

        forecast_years:
            Number of future years to display.

        history_years:
            Number of recent historical years to display.
            When None, the complete historical series is shown.

        title:
            Primary chart title.

        subtitle:
            Optional subtitle. When omitted, one is generated
            from the forecast period and growth assumption.

        show_disclaimer:
            Whether to display the forecast disclaimer.

        show_projected_value:
            Whether to annotate the final projected value.

        source_note:
            Optional source or attribution text displayed along
            the bottom of the Figure.

        watermark:
            Optional subtle watermark displayed inside the Axes.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Figure and Axes.
    """

    if history_years is not None:
        if (
            isinstance(history_years, bool)
            or not isinstance(history_years, int)
        ):
            raise TypeError(
                "history_years must be an integer or None."
            )

        if history_years < 1:
            raise ValueError(
                "history_years must be at least 1."
            )

    if not isinstance(
        show_disclaimer,
        bool,
    ):
        raise TypeError(
            "show_disclaimer must be a Boolean value."
        )

    if not isinstance(
        show_projected_value,
        bool,
    ):
        raise TypeError(
            "show_projected_value must be a Boolean value."
        )

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    valid_prices = (
        prices
        .dropna()
        .sort_index()
    )

    if valid_prices.empty:
        raise ValueError(
            "No valid historical prices are available "
            "for the forecast chart."
        )

    if history_years is None:
        visible_history = valid_prices
    else:
        history_start_date = (
            valid_prices.index[-1]
            - pd.DateOffset(
                years=history_years
            )
        )

        visible_history = valid_prices.loc[
            valid_prices.index
            >= history_start_date
        ]

    forecast = generate_forecast_series(
        price_series=valid_prices,
        annual_growth_rate=annual_growth_rate,
        years=forecast_years,
    )

    forecast_boundary = valid_prices.index[-1]

    connected_forecast = pd.concat(
        [
            valid_prices.iloc[-1:].rename(
                forecast.name
            ),
            forecast,
        ]
    )

    resolved_subtitle = (
        subtitle
        if subtitle is not None
        else DEFAULT_FORECAST_SUBTITLE.format(
            forecast_years=forecast_years,
            growth_rate=annual_growth_rate,
        )
    )

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        visible_history.index,
        visible_history.values,
        color=HISTORICAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_FORECAST_HISTORY_LABEL,
    )

    axes.plot(
        connected_forecast.index,
        connected_forecast.values,
        color=FORECAST_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        linestyle=DEFAULT_FORECAST_LINE_STYLE,
        marker="o",
        markersize=DEFAULT_FORECAST_MARKER_SIZE,
        markevery=[
            len(connected_forecast) - 1
        ],
        label=(
            f"{DEFAULT_FORECAST_LABEL} "
            f"({annual_growth_rate:.1%} annually)"
        ),
    )

    annotate_vertical_event(
        axes,
        x_value=forecast_boundary,
        label=DEFAULT_FORECAST_BOUNDARY_LABEL,
        line_color=FORECAST_BOUNDARY_COLOR,
        line_style=DEFAULT_FORECAST_BOUNDARY_STYLE,
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=title,
        subtitle=resolved_subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_FORECAST_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_currency_axis(
        axes
    )

    if show_projected_value:
        annotate_latest_value(
            axes,
            forecast,
            prefix=(
                f"Projected {forecast_years}-Year Value"
            ),
            value_format="currency",
            marker_color=FORECAST_COLOR,
            offset=(
                -150,
                16,
            ),
        )

    if watermark is not None:
        add_watermark(
            axes,
            watermark,
        )

    if show_disclaimer:
        figure.text(
            0.01,
            0.01,
            DEFAULT_FORECAST_DISCLAIMER,
            color=SECONDARY_TEXT_COLOR,
            fontsize=8,
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    source_note_y_position = (
        0.028
        if show_disclaimer
        else 0.01
    )

    if source_note is not None:
        add_figure_source_note(
            figure,
            source_note,
            y_position=source_note_y_position,
        )

    bottom_margin = 0.0

    if show_disclaimer:
        bottom_margin = 0.04

    if source_note is not None:
        bottom_margin = 0.07 if show_disclaimer else 0.04

    figure.tight_layout(
        pad=1.5,
        rect=(
            0.0,
            bottom_margin,
            1.0,
            1.0,
        ),
    )

    return figure, axes


def plot_forecast_scenarios(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    *,
    growth_rates: dict[str, float] | None = None,
    forecast_years: int = (
        DEFAULT_FORECAST_PERIOD_YEARS
    ),
    history_years: int | None = (
        DEFAULT_FORECAST_HISTORY_YEARS
    ),
    title: str = DEFAULT_FORECAST_SCENARIO_TITLE,
    subtitle: str | None = None,
    show_disclaimer: bool = True,
    source_note: str | None = None,
    watermark: str | None = None,
    figure_size: tuple[float, float] = (
        DEFAULT_FIGURE_SIZE
    ),
) -> tuple[Figure, Axes]:
    """
    Plot historical prices followed by multiple hypothetical
    forecast scenarios.

    Args:
        data:
            Source DataFrame containing historical prices.

        column:
            Numeric price column used for the chart.

        growth_rates:
            Mapping of scenario names to annual growth rates.
            When omitted, the default scenarios are used.

        forecast_years:
            Number of future years to display.

        history_years:
            Number of recent historical years to display.
            When None, the complete historical series is shown.

        title:
            Primary chart title.

        subtitle:
            Optional subtitle. When omitted, one is generated
            from the forecast horizon.

        show_disclaimer:
            Whether to display the forecast disclaimer.

        source_note:
            Optional source or attribution text displayed along
            the bottom of the Figure.

        watermark:
            Optional subtle watermark displayed inside the Axes.

        figure_size:
            Matplotlib figure dimensions as (width, height).

    Returns:
        A tuple containing the styled Figure and Axes.
    """

    if history_years is not None:
        if (
            isinstance(history_years, bool)
            or not isinstance(history_years, int)
        ):
            raise TypeError(
                "history_years must be an integer or None."
            )

        if history_years < 1:
            raise ValueError(
                "history_years must be at least 1."
            )

    if not isinstance(
        show_disclaimer,
        bool,
    ):
        raise TypeError(
            "show_disclaimer must be a Boolean value."
        )

    prices = prepare_price_series(
        data=data,
        column=column,
    )

    valid_prices = (
        prices
        .dropna()
        .sort_index()
    )

    if valid_prices.empty:
        raise ValueError(
            "No valid historical prices are available "
            "for the scenario chart."
        )

    if history_years is None:
        visible_history = valid_prices
    else:
        history_start_date = (
            valid_prices.index[-1]
            - pd.DateOffset(
                years=history_years
            )
        )

        visible_history = valid_prices.loc[
            valid_prices.index
            >= history_start_date
        ]

    scenario_forecasts = generate_forecast_scenarios(
        price_series=valid_prices,
        growth_rates=growth_rates,
        years=forecast_years,
    )

    resolved_growth_rates = (
        DEFAULT_FORECAST_SCENARIOS
        if growth_rates is None
        else growth_rates
    )

    resolved_subtitle = (
        subtitle
        if subtitle is not None
        else DEFAULT_FORECAST_SCENARIO_SUBTITLE.format(
            forecast_years=forecast_years
        )
    )

    forecast_boundary = valid_prices.index[-1]

    figure, axes = create_figure(
        figure_size=figure_size,
    )

    axes.plot(
        visible_history.index,
        visible_history.values,
        color=HISTORICAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DEFAULT_FORECAST_HISTORY_LABEL,
    )

    default_scenario_colors = {
        "Conservative": (
            FORECAST_CONSERVATIVE_COLOR
        ),
        "Expected": FORECAST_BASE_COLOR,
        "Optimistic": (
            FORECAST_OPTIMISTIC_COLOR
        ),
    }

    for scenario_index, scenario_name in enumerate(
        scenario_forecasts.columns
    ):
        scenario_series = scenario_forecasts[
            scenario_name
        ]

        connected_scenario = pd.concat(
            [
                valid_prices.iloc[-1:].rename(
                    scenario_name
                ),
                scenario_series,
            ]
        )

        growth_rate = resolved_growth_rates[
            scenario_name
        ]

        scenario_color = (
            default_scenario_colors.get(
                scenario_name,
                SCENARIO_COLORS[
                    scenario_index
                    % len(SCENARIO_COLORS)
                ],
            )
        )

        axes.plot(
            connected_scenario.index,
            connected_scenario.values,
            color=scenario_color,
            linewidth=DEFAULT_LINE_WIDTH,
            linestyle=DEFAULT_FORECAST_LINE_STYLE,
            marker="o",
            markersize=(
                DEFAULT_FORECAST_MARKER_SIZE
            ),
            markevery=[
                len(connected_scenario) - 1
            ],
            label=(
                f"{scenario_name} "
                f"({growth_rate:.1%})"
            ),
        )

    annotate_vertical_event(
        axes,
        x_value=forecast_boundary,
        label=DEFAULT_FORECAST_BOUNDARY_LABEL,
        line_color=FORECAST_BOUNDARY_COLOR,
        line_style=DEFAULT_FORECAST_BOUNDARY_STYLE,
    )

    axes.xaxis.set_major_locator(
        mdates.AutoDateLocator()
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            axes.xaxis.get_major_locator()
        )
    )

    apply_standard_formatting(
        figure,
        axes,
        title=title,
        subtitle=resolved_subtitle,
        x_label=DEFAULT_X_AXIS_LABEL,
        y_label=DEFAULT_FORECAST_Y_AXIS_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_currency_axis(
        axes
    )

    if watermark is not None:
        add_watermark(
            axes,
            watermark,
        )

    if show_disclaimer:
        figure.text(
            0.01,
            0.01,
            DEFAULT_FORECAST_DISCLAIMER,
            color=SECONDARY_TEXT_COLOR,
            fontsize=8,
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    source_note_y_position = (
        0.028
        if show_disclaimer
        else 0.01
    )

    if source_note is not None:
        add_figure_source_note(
            figure,
            source_note,
            y_position=source_note_y_position,
        )

    bottom_margin = 0.0

    if show_disclaimer:
        bottom_margin = 0.04

    if source_note is not None:
        bottom_margin = 0.07 if show_disclaimer else 0.04

    figure.tight_layout(
        pad=1.5,
        rect=(
            0.0,
            bottom_margin,
            1.0,
            1.0,
        ),
    )

    return figure, axes