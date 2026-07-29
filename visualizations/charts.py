"""
Reusable chart functions for the Gold Value Predictor.

Each plotting function returns its Matplotlib Figure and Axes
objects instead of displaying the chart automatically. This
allows charts to be tested, customized, displayed, or saved by
other parts of the application.
"""

from pathlib import Path
from matplotlib.ticker import FuncFormatter
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from analytics.returns import (
    DEFAULT_PRICE_COLUMN,
    prepare_price_series,
)

from analytics.rolling import (
    DEFAULT_ROLLING_WINDOW,
    calculate_rolling_average,
)

from analytics.returns import (
    DEFAULT_PRICE_COLUMN,
    calculate_monthly_returns,
    prepare_price_series,
)

from analytics.rolling import (
    DEFAULT_ROLLING_WINDOW,
    calculate_rolling_average,
    calculate_rolling_volatility,
)

from analytics.rolling import (
    DEFAULT_ROLLING_WINDOW,
    calculate_rolling_average,
    calculate_rolling_return,
    calculate_rolling_volatility,
)

from analytics.rolling import (
    DEFAULT_ROLLING_WINDOW,
    calculate_rolling_average,
    calculate_rolling_high,
    calculate_rolling_low,
    calculate_rolling_return,
    calculate_rolling_volatility,
)

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
DEFAULT_RECENT_YEARS = 5
DEFAULT_RECENT_PRICE_TITLE = "Recent Gold Prices"
DEFAULT_INFLATION_ADJUSTED_COLUMN = "Inflation_Adjusted_Price"
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

DEFAULT_REAL_LABEL = "Inflation-Adjusted Gold Price"

DEFAULT_COMPARISON_Y_AXIS_LABEL = (
    "Gold Price (USD per Troy Ounce)"
)

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

def validate_figure_size(
    figure_size: tuple[float, float],
) -> tuple[float, float]:
    """
    Validate and normalize a Matplotlib figure size.

    Args:
        figure_size:
            Two-item tuple containing the width and height in
            inches.

    Returns:
        The validated width and height as floats.

    Raises:
        TypeError:
            If figure_size is not a tuple containing exactly
            two numeric values.

        ValueError:
            If either dimension is less than or equal to zero.
    """

    if not isinstance(figure_size, tuple):
        raise TypeError(
            "figure_size must be a tuple containing width "
            "and height."
        )

    if len(figure_size) != 2:
        raise TypeError(
            "figure_size must contain exactly two values."
        )

    width, height = figure_size

    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, (int, float))
        or not isinstance(height, (int, float))
    ):
        raise TypeError(
            "figure_size width and height must be numeric."
        )

    normalized_width = float(width)
    normalized_height = float(height)

    if normalized_width <= 0 or normalized_height <= 0:
        raise ValueError(
            "figure_size width and height must be greater "
            "than zero."
        )

    return normalized_width, normalized_height


def validate_line_width(
    line_width: float,
) -> float:
    """
    Validate a Matplotlib line width.

    Args:
        line_width:
            Width of the plotted line.

    Returns:
        The validated line width as a float.

    Raises:
        TypeError:
            If line_width is not numeric.

        ValueError:
            If line_width is less than or equal to zero.
    """

    if (
        isinstance(line_width, bool)
        or not isinstance(line_width, (int, float))
    ):
        raise TypeError(
            "line_width must be numeric."
        )

    normalized_line_width = float(line_width)

    if normalized_line_width <= 0:
        raise ValueError(
            "line_width must be greater than zero."
        )

    return normalized_line_width


def validate_text_option(
    value: str,
    option_name: str,
) -> str:
    """
    Validate a required chart text option.

    Args:
        value:
            Text to validate.

        option_name:
            Name of the option for error messages.

    Returns:
        The stripped text value.

    Raises:
        TypeError:
            If value is not a string.

        ValueError:
            If value is empty or contains only whitespace.
    """

    if not isinstance(value, str):
        raise TypeError(
            f"{option_name} must be a string."
        )

    normalized_value = value.strip()

    if not normalized_value:
        raise ValueError(
            f"{option_name} cannot be empty."
        )

    return normalized_value


def validate_optional_text(
    value: str | None,
    option_name: str,
) -> str | None:
    """
    Validate an optional chart text value.

    Args:
        value:
            Optional text to validate.

        option_name:
            Name of the option for error messages.

    Returns:
        None or the stripped text value.

    Raises:
        TypeError:
            If value is neither a string nor None.

        ValueError:
            If value is a blank string.
    """

    if value is None:
        return None

    return validate_text_option(
        value=value,
        option_name=option_name,
    )


def validate_boolean_option(
    value: bool,
    option_name: str,
) -> bool:
    """
    Validate a boolean chart option.
    """

    if not isinstance(value, bool):
        raise TypeError(
            f"{option_name} must be a boolean."
        )

    return value


def format_currency_axis(
    value: float,
    _position: float,
) -> str:
    """
    Format a numeric axis value as U.S. currency.

    Whole-dollar labels are used for values of at least one
    dollar. Smaller values retain two decimal places.
    """

    if abs(value) < 1:
        return f"${value:,.2f}"

    return f"${value:,.0f}"


def format_currency_value(
    value: float,
) -> str:
    """
    Format a price value as U.S. currency.
    """

    return f"${value:,.2f}"


def create_date_range_subtitle(
    prices: pd.Series,
) -> str:
    """
    Create a readable subtitle describing the data period.

    Args:
        prices:
            Chronological price Series with a DatetimeIndex.

    Returns:
        A subtitle containing the first date, final date, and
        number of observations.
    """

    start_date = prices.index[0].strftime("%b %Y")
    end_date = prices.index[-1].strftime("%b %Y")

    observation_count = len(prices)

    observation_label = (
        "observation"
        if observation_count == 1
        else "observations"
    )

    return (
        f"{start_date} through {end_date}"
        f"  |  {observation_count:,} {observation_label}"
    )


def format_date_axis(
    axes: Axes,
    dates: pd.DatetimeIndex,
) -> None:
    """
    Apply date locators and formatters appropriate for the
    duration of a chart.

    Short periods use monthly labels. Medium periods use
    annual labels. Very long periods use evenly spaced year
    intervals to prevent overcrowding.
    """

    if not isinstance(axes, Axes):
        raise TypeError(
            "axes must be a Matplotlib Axes."
        )

    if not isinstance(dates, pd.DatetimeIndex):
        raise TypeError(
            "dates must be a pandas DatetimeIndex."
        )

    if dates.empty:
        raise ValueError(
            "dates cannot be empty."
        )

    total_days = (
        dates.max() - dates.min()
    ).days

    approximate_years = (
        total_days / 365.2425
    )

    if approximate_years <= 2:
        major_locator = mdates.MonthLocator(
            interval=3
        )

        major_formatter = mdates.DateFormatter(
            "%b\n%Y"
        )

    elif approximate_years <= 15:
        major_locator = mdates.YearLocator(
            base=1
        )

        major_formatter = mdates.DateFormatter(
            "%Y"
        )

    elif approximate_years <= 50:
        major_locator = mdates.YearLocator(
            base=5
        )

        major_formatter = mdates.DateFormatter(
            "%Y"
        )

    else:
        major_locator = mdates.YearLocator(
            base=20
        )

        major_formatter = mdates.DateFormatter(
            "%Y"
        )

    axes.xaxis.set_major_locator(
        major_locator
    )

    axes.xaxis.set_major_formatter(
        major_formatter
    )

    axes.tick_params(
        axis="x",
        rotation=0,
    )

    axes.margins(
        x=0.01
    )


def apply_chart_layout(
    axes: Axes,
    show_grid: bool = True,
) -> None:
    """
    Apply shared visual formatting to a chart.

    The top and right borders are hidden, the remaining borders
    are softened, and horizontal grid lines are used when grid
    display is enabled.
    """

    if not isinstance(axes, Axes):
        raise TypeError(
            "axes must be a Matplotlib Axes."
        )

    normalized_show_grid = validate_boolean_option(
        value=show_grid,
        option_name="show_grid",
    )

    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)

    axes.spines["left"].set_alpha(0.5)
    axes.spines["bottom"].set_alpha(0.5)

    axes.tick_params(
        axis="both",
        which="major",
        labelsize=10,
    )

    axes.set_axisbelow(True)

    axes.grid(
        visible=normalized_show_grid,
        axis="y",
        linestyle="--",
        linewidth=0.8,
        alpha=DEFAULT_GRID_ALPHA,
    )

    axes.grid(
        visible=False,
        axis="x",
    )


def add_latest_value_annotation(
    axes: Axes,
    prices: pd.Series,
) -> None:
    """
    Mark and label the most recent price observation.

    Args:
        axes:
            Axes containing the historical-price chart.

        prices:
            Chronological price Series.
    """

    if not isinstance(axes, Axes):
        raise TypeError(
            "axes must be a Matplotlib Axes."
        )

    if not isinstance(prices, pd.Series):
        raise TypeError(
            "prices must be a pandas Series."
        )

    if prices.empty:
        raise ValueError(
            "prices cannot be empty."
        )

    latest_date = prices.index[-1]
    latest_price = float(prices.iloc[-1])

    axes.scatter(
        [latest_date],
        [latest_price],
        zorder=3,
        label="_nolegend_",
    )

    axes.annotate(
        format_currency_value(latest_price),
        xy=(
            latest_date,
            latest_price,
        ),
        xytext=DEFAULT_ANNOTATION_OFFSET,
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        arrowprops={
            "arrowstyle": "-",
            "linewidth": 0.8,
            "alpha": 0.6,
        },
        annotation_clip=False,
    )


def add_chart_titles(
    axes: Axes,
    title: str,
    subtitle: str | None = None,
) -> None:
    """
    Add a main chart title and optional subtitle.
    """

    if not isinstance(axes, Axes):
        raise TypeError(
            "axes must be a Matplotlib Axes."
        )

    normalized_title = validate_text_option(
        value=title,
        option_name="title",
    )

    normalized_subtitle = validate_optional_text(
        value=subtitle,
        option_name="subtitle",
    )

    axes.set_title(
        normalized_title,
        loc="left",
        fontsize=16,
        fontweight="bold",
        pad=DEFAULT_TITLE_PADDING,
    )

    if normalized_subtitle is not None:
        axes.text(
            0.0,
            1.02,
            normalized_subtitle,
            transform=axes.transAxes,
            fontsize=10,
            alpha=0.7,
            ha="left",
            va="bottom",
        )

def _plot_price_series(
    prices: pd.Series,
    *,
    price_label: str,
    title: str,
    subtitle: str | None,
    x_axis_label: str,
    y_axis_label: str,
    figure_size: tuple[float, float],
    line_width: float,
    show_grid: bool,
    show_latest_value: bool,
    show_legend: bool,
) -> tuple[Figure, Axes]:
    """
    Create a consistently styled chart from a prepared price
    Series.

    This is an internal helper used by public price-chart
    functions. The supplied Series is expected to have already
    been validated and prepared.
    """

    if not isinstance(prices, pd.Series):
        raise TypeError(
            "prices must be a pandas Series."
        )

    if prices.empty:
        raise ValueError(
            "prices cannot be empty."
        )

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(
            "prices must have a pandas DatetimeIndex."
        )

    normalized_price_label = validate_text_option(
        value=price_label,
        option_name="price_label",
    )

    normalized_title = validate_text_option(
        value=title,
        option_name="title",
    )

    normalized_subtitle = validate_optional_text(
        value=subtitle,
        option_name="subtitle",
    )

    normalized_x_label = validate_text_option(
        value=x_axis_label,
        option_name="x_axis_label",
    )

    normalized_y_label = validate_text_option(
        value=y_axis_label,
        option_name="y_axis_label",
    )

    normalized_figure_size = validate_figure_size(
        figure_size=figure_size
    )

    normalized_line_width = validate_line_width(
        line_width=line_width
    )

    normalized_show_grid = validate_boolean_option(
        value=show_grid,
        option_name="show_grid",
    )

    normalized_show_latest_value = (
        validate_boolean_option(
            value=show_latest_value,
            option_name="show_latest_value",
        )
    )

    normalized_show_legend = validate_boolean_option(
        value=show_legend,
        option_name="show_legend",
    )

    if normalized_subtitle is None:
        normalized_subtitle = create_date_range_subtitle(
            prices=prices
        )

    figure, axes = plt.subplots(
        figsize=normalized_figure_size
    )

    axes.plot(
        prices.index,
        prices.values,
        linewidth=normalized_line_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=normalized_price_label,
    )

    add_chart_titles(
        axes=axes,
        title=normalized_title,
        subtitle=normalized_subtitle,
    )

    axes.set_xlabel(
        normalized_x_label,
        fontsize=11,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.set_ylabel(
        normalized_y_label,
        fontsize=11,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.yaxis.set_major_formatter(
        FuncFormatter(format_currency_axis)
    )

    format_date_axis(
        axes=axes,
        dates=prices.index,
    )

    apply_chart_layout(
        axes=axes,
        show_grid=normalized_show_grid,
    )

    if normalized_show_latest_value:
        add_latest_value_annotation(
            axes=axes,
            prices=prices,
        )

    if normalized_show_legend:
        axes.legend(
            loc="upper left",
            frameon=False,
            fontsize=10,
        )

    figure.tight_layout(
        pad=1.5
    )

    return figure, axes

def plot_historical_price(
    data: pd.DataFrame,
    price_column: str = DEFAULT_PRICE_COLUMN,
    title: str = DEFAULT_CHART_TITLE,
    subtitle: str | None = None,
    x_axis_label: str = DEFAULT_X_AXIS_LABEL,
    y_axis_label: str = DEFAULT_Y_AXIS_LABEL,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    show_grid: bool = True,
    show_latest_value: bool = True,
    show_legend: bool = True,
) -> tuple[Figure, Axes]:
    """
    Create a polished chart of the complete price history.

    The function does not call plt.show(). The caller decides
    whether to display, save, customize, or close the returned
    figure.

    Args:
        data:
            DataFrame containing a Date column and the requested
            price column.

        price_column:
            Name of the numeric price column to chart.

        title:
            Main chart title.

        subtitle:
            Optional subtitle. When omitted, a date-range
            subtitle is generated automatically.

        x_axis_label:
            Label displayed beneath the horizontal axis.

        y_axis_label:
            Label displayed beside the vertical axis.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of the historical-price line.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_value:
            Whether to mark and label the most recent price.

        show_legend:
            Whether to display the chart legend.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.
    """

    prices = prepare_price_series(
        data=data,
        column=price_column,
    )

    return _plot_price_series(
        prices=prices,
        price_label=price_column,
        title=title,
        subtitle=subtitle,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        figure_size=figure_size,
        line_width=line_width,
        show_grid=show_grid,
        show_latest_value=show_latest_value,
        show_legend=show_legend,
    )

def plot_recent_price(
    data: pd.DataFrame,
    years: int = DEFAULT_RECENT_YEARS,
    price_column: str = DEFAULT_PRICE_COLUMN,
    title: str | None = None,
    subtitle: str | None = None,
    x_axis_label: str = DEFAULT_X_AXIS_LABEL,
    y_axis_label: str = DEFAULT_Y_AXIS_LABEL,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    show_grid: bool = True,
    show_latest_value: bool = True,
    show_legend: bool = True,
) -> tuple[Figure, Axes]:
    """
    Create a polished chart of the most recent price history.

    The requested date window is measured backward from the
    latest observation in the dataset. For example, years=5
    produces a chart covering approximately the final five
    calendar years of available data.

    Args:
        data:
            DataFrame containing a Date column and the requested
            price column.

        years:
            Number of recent calendar years to display.

        price_column:
            Name of the numeric price column to chart.

        title:
            Optional main chart title. When omitted, a title is
            generated from the requested number of years.

        subtitle:
            Optional subtitle. When omitted, a date-range
            subtitle is generated automatically.

        x_axis_label:
            Label displayed beneath the horizontal axis.

        y_axis_label:
            Label displayed beside the vertical axis.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of the plotted line.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_value:
            Whether to mark and label the most recent price.

        show_legend:
            Whether to display the chart legend.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If years is not an integer or another input has an
            invalid type.

        ValueError:
            If years is less than one or the dataset cannot
            produce a valid recent price series.
    """

    normalized_years = validate_years(
        years=years
    )

    prices = prepare_price_series(
        data=data,
        column=price_column,
    )

    recent_prices = filter_recent_years(
        prices=prices,
        years=normalized_years,
    )

    if title is None:
        year_label = (
            "Year"
            if normalized_years == 1
            else "Years"
        )

        normalized_title = (
            f"{DEFAULT_RECENT_PRICE_TITLE} "
            f"(Last {normalized_years} {year_label})"
        )
    else:
        normalized_title = validate_text_option(
            value=title,
            option_name="title",
        )

    return _plot_price_series(
        prices=recent_prices,
        price_label=price_column,
        title=normalized_title,
        subtitle=subtitle,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        figure_size=figure_size,
        line_width=line_width,
        show_grid=show_grid,
        show_latest_value=show_latest_value,
        show_legend=show_legend,
    )

def plot_inflation_adjusted_price(
    data: pd.DataFrame,
    inflation_adjusted_column: str = (
        DEFAULT_INFLATION_ADJUSTED_COLUMN
    ),
    title: str = DEFAULT_INFLATION_ADJUSTED_TITLE,
    subtitle: str | None = None,
    x_axis_label: str = DEFAULT_X_AXIS_LABEL,
    y_axis_label: str = (
        DEFAULT_INFLATION_ADJUSTED_Y_AXIS_LABEL
    ),
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    show_grid: bool = True,
    show_latest_value: bool = True,
    show_legend: bool = True,
) -> tuple[Figure, Axes]:
    """
    Create a polished chart of inflation-adjusted gold prices.

    This function expects the supplied DataFrame to already
    contain an inflation-adjusted price column. Calculating the
    inflation-adjusted values remains the responsibility of the
    analytics layer.

    The function does not call plt.show(). The caller decides
    whether to display, save, customize, or close the returned
    figure.

    Args:
        data:
            DataFrame containing a Date column and an
            inflation-adjusted price column.

        inflation_adjusted_column:
            Name of the column containing inflation-adjusted
            gold prices.

        title:
            Main chart title.

        subtitle:
            Optional subtitle. When omitted, a date-range
            subtitle is generated automatically.

        x_axis_label:
            Label displayed beneath the horizontal axis.

        y_axis_label:
            Label displayed beside the vertical axis.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of the plotted line.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_value:
            Whether to mark and label the most recent value.

        show_legend:
            Whether to display the chart legend.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If an argument has an invalid type.

        ValueError:
            If the requested price series cannot be prepared.

        KeyError:
            If the inflation-adjusted column does not exist.
    """

    normalized_column = validate_text_option(
        value=inflation_adjusted_column,
        option_name="inflation_adjusted_column",
    )

    if normalized_column not in data.columns:
        available_columns = ", ".join(
            str(column)
            for column in data.columns
        )

        raise KeyError(
            f"Column '{normalized_column}' was not found. "
            f"Available columns: {available_columns}"
        )

    prices = prepare_price_series(
        data=data,
        column=normalized_column,
    )

    return _plot_price_series(
        prices=prices,
        price_label=DEFAULT_INFLATION_ADJUSTED_LABEL,
        title=title,
        subtitle=subtitle,
        x_axis_label=x_axis_label,
        y_axis_label=y_axis_label,
        figure_size=figure_size,
        line_width=line_width,
        show_grid=show_grid,
        show_latest_value=show_latest_value,
        show_legend=show_legend,
    )

def plot_nominal_vs_inflation_adjusted(
    data: pd.DataFrame,
    nominal_column: str = DEFAULT_PRICE_COLUMN,
    inflation_adjusted_column: str = (
        DEFAULT_INFLATION_ADJUSTED_COLUMN
    ),
    title: str = DEFAULT_NOMINAL_VS_REAL_TITLE,
    subtitle: str | None = None,
    nominal_label: str = DEFAULT_NOMINAL_LABEL,
    inflation_adjusted_label: str = DEFAULT_REAL_LABEL,
    x_axis_label: str = DEFAULT_X_AXIS_LABEL,
    y_axis_label: str = DEFAULT_COMPARISON_Y_AXIS_LABEL,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    show_grid: bool = True,
    show_latest_values: bool = True,
    show_legend: bool = True,
) -> tuple[Figure, Axes]:
    """
    Compare nominal and inflation-adjusted gold prices.

    The supplied DataFrame must contain a Date column, a
    nominal-price column, and an inflation-adjusted-price
    column. Inflation calculations remain the responsibility
    of the analytics layer.

    The function does not call plt.show(). The caller decides
    whether to display, save, customize, or close the returned
    figure.

    Args:
        data:
            DataFrame containing dates, nominal prices, and
            inflation-adjusted prices.

        nominal_column:
            Name of the nominal gold-price column.

        inflation_adjusted_column:
            Name of the inflation-adjusted price column.

        title:
            Main chart title.

        subtitle:
            Optional subtitle. When omitted, a date-range
            subtitle is generated automatically.

        nominal_label:
            Legend label for the nominal-price line.

        inflation_adjusted_label:
            Legend label for the inflation-adjusted line.

        x_axis_label:
            Label displayed beneath the horizontal axis.

        y_axis_label:
            Label displayed beside the vertical axis.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of both plotted lines.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_values:
            Whether to label the latest value of both series.

        show_legend:
            Whether to display the chart legend.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If an argument has an invalid type.

        ValueError:
            If no overlapping nominal and adjusted prices are
            available.

        KeyError:
            If a requested column is unavailable.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    normalized_nominal_column = validate_text_option(
        value=nominal_column,
        option_name="nominal_column",
    )

    normalized_adjusted_column = validate_text_option(
        value=inflation_adjusted_column,
        option_name="inflation_adjusted_column",
    )

    normalized_title = validate_text_option(
        value=title,
        option_name="title",
    )

    normalized_subtitle = validate_optional_text(
        value=subtitle,
        option_name="subtitle",
    )

    normalized_nominal_label = validate_text_option(
        value=nominal_label,
        option_name="nominal_label",
    )

    normalized_adjusted_label = validate_text_option(
        value=inflation_adjusted_label,
        option_name="inflation_adjusted_label",
    )

    normalized_x_label = validate_text_option(
        value=x_axis_label,
        option_name="x_axis_label",
    )

    normalized_y_label = validate_text_option(
        value=y_axis_label,
        option_name="y_axis_label",
    )

    normalized_figure_size = validate_figure_size(
        figure_size=figure_size
    )

    normalized_line_width = validate_line_width(
        line_width=line_width
    )

    normalized_show_grid = validate_boolean_option(
        value=show_grid,
        option_name="show_grid",
    )

    normalized_show_latest_values = (
        validate_boolean_option(
            value=show_latest_values,
            option_name="show_latest_values",
        )
    )

    normalized_show_legend = validate_boolean_option(
        value=show_legend,
        option_name="show_legend",
    )

    missing_columns = [
        column
        for column in (
            normalized_nominal_column,
            normalized_adjusted_column,
        )
        if column not in data.columns
    ]

    if missing_columns:
        missing_text = ", ".join(missing_columns)

        available_columns = ", ".join(
            str(column)
            for column in data.columns
        )

        raise KeyError(
            f"Required column(s) not found: {missing_text}. "
            f"Available columns: {available_columns}"
        )

    nominal_prices = prepare_price_series(
        data=data,
        column=normalized_nominal_column,
    )

    adjusted_prices = prepare_price_series(
        data=data,
        column=normalized_adjusted_column,
    )

    comparison_data = pd.concat(
        [
            nominal_prices.rename(
                normalized_nominal_column
            ),
            adjusted_prices.rename(
                normalized_adjusted_column
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if comparison_data.empty:
        raise ValueError(
            "No overlapping nominal and inflation-adjusted "
            "price observations are available."
        )

    comparison_data = comparison_data.sort_index()

    nominal_prices = comparison_data[
        normalized_nominal_column
    ]

    adjusted_prices = comparison_data[
        normalized_adjusted_column
    ]

    if normalized_subtitle is None:
        normalized_subtitle = create_date_range_subtitle(
            prices=nominal_prices
        )

    figure, axes = plt.subplots(
        figsize=normalized_figure_size
    )

    axes.plot(
        nominal_prices.index,
        nominal_prices.values,
        linewidth=normalized_line_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=normalized_nominal_label,
    )

    axes.plot(
        adjusted_prices.index,
        adjusted_prices.values,
        linewidth=normalized_line_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=normalized_adjusted_label,
    )

    add_chart_titles(
        axes=axes,
        title=normalized_title,
        subtitle=normalized_subtitle,
    )

    axes.set_xlabel(
        normalized_x_label,
        fontsize=11,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.set_ylabel(
        normalized_y_label,
        fontsize=11,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.yaxis.set_major_formatter(
        FuncFormatter(format_currency_axis)
    )

    format_date_axis(
        axes=axes,
        dates=comparison_data.index,
    )

    apply_chart_layout(
        axes=axes,
        show_grid=normalized_show_grid,
    )

    if normalized_show_latest_values:
        add_latest_value_annotation(
            axes=axes,
            prices=nominal_prices,
        )

        add_latest_value_annotation(
            axes=axes,
            prices=adjusted_prices,
        )

    if normalized_show_legend:
        axes.legend(
            loc="upper left",
            frameon=False,
            fontsize=10,
        )

    figure.tight_layout(
        pad=1.5
    )

    return figure, axes

def save_chart(
    figure: Figure,
    output_path: str | Path,
    dpi: int = DEFAULT_DPI,
) -> Path:
    """
    Save a Matplotlib figure to disk.

    Parent directories are created automatically when needed.

    Args:
        figure:
            Matplotlib Figure to save.

        output_path:
            Destination path, including the file name and
            extension.

        dpi:
            Image resolution in dots per inch.

    Returns:
        The resolved path of the saved chart.

    Raises:
        TypeError:
            If figure is not a Matplotlib Figure, output_path is
            invalid, or dpi is not an integer.

        ValueError:
            If output_path is empty, has no file extension, or
            dpi is less than one.
    """

    if not isinstance(figure, Figure):
        raise TypeError(
            "figure must be a Matplotlib Figure."
        )

    if not isinstance(output_path, (str, Path)):
        raise TypeError(
            "output_path must be a string or Path."
        )

    if isinstance(output_path, str) and not output_path.strip():
        raise ValueError(
            "output_path cannot be empty."
        )

    if isinstance(dpi, bool) or not isinstance(dpi, int):
        raise TypeError(
            "dpi must be an integer."
        )

    if dpi < 1:
        raise ValueError(
            "dpi must be at least 1."
        )

    path = Path(output_path).expanduser()

    if not path.suffix:
        raise ValueError(
            "output_path must include a file extension."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    return path.resolve()

def validate_years(
    years: int,
) -> int:
    """
    Validate a number of calendar years.

    Args:
        years:
            Positive whole number of years.

    Returns:
        The validated number of years.

    Raises:
        TypeError:
            If years is not an integer.

        ValueError:
            If years is less than one.
    """

    if isinstance(years, bool) or not isinstance(years, int):
        raise TypeError(
            "years must be an integer."
        )

    if years < 1:
        raise ValueError(
            "years must be at least 1."
        )

    return years


def filter_recent_years(
    prices: pd.Series,
    years: int = DEFAULT_RECENT_YEARS,
) -> pd.Series:
    """
    Filter a chronological price Series to its most recent
    number of calendar years.

    The filtering window is measured backward from the final
    date in the Series rather than from today's date. This keeps
    the function reliable for historical and test datasets.

    Args:
        prices:
            Price Series with a DatetimeIndex.

        years:
            Number of calendar years to retain.

    Returns:
        A copy of the filtered price Series.

    Raises:
        TypeError:
            If prices is not a pandas Series, its index is not a
            DatetimeIndex, or years is not an integer.

        ValueError:
            If prices is empty, years is less than one, or the
            filtering operation produces no observations.
    """

    if not isinstance(prices, pd.Series):
        raise TypeError(
            "prices must be a pandas Series."
        )

    if prices.empty:
        raise ValueError(
            "prices cannot be empty."
        )

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise TypeError(
            "prices must have a pandas DatetimeIndex."
        )

    normalized_years = validate_years(
        years=years
    )

    chronological_prices = prices.sort_index()

    end_date = chronological_prices.index.max()

    start_date = end_date - pd.DateOffset(
        years=normalized_years
    )

    recent_prices = chronological_prices.loc[
        chronological_prices.index >= start_date
    ].copy()

    if recent_prices.empty:
        raise ValueError(
            "No price observations were found within the "
            "requested date range."
        )

    return recent_prices

def _plot_rolling_series(
    prices: pd.Series,
    rolling_series: pd.Series,
    title: str,
    subtitle: str | None,
    price_label: str,
    rolling_label: str,
    x_axis_label: str,
    y_axis_label: str,
    figure_size: tuple[float, float],
    line_width: float,
    show_grid: bool,
    show_latest_values: bool,
    show_legend: bool,
) -> tuple[Figure, Axes]:
    """
    Plot a source price Series alongside a rolling metric.

    This private helper handles shared presentation logic for
    rolling-price charts. The rolling calculation itself must
    be completed by the analytics layer before this function
    is called.

    Args:
        prices:
            Prepared source-price Series indexed by date.

        rolling_series:
            Prepared rolling metric indexed by date.

        title:
            Main chart title.

        subtitle:
            Optional subtitle.

        price_label:
            Legend label for the source-price line.

        rolling_label:
            Legend label for the rolling-metric line.

        x_axis_label:
            Horizontal-axis label.

        y_axis_label:
            Vertical-axis label.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of the plotted lines.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_values:
            Whether to annotate the latest valid values.

        show_legend:
            Whether to display the chart legend.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        ValueError:
            If either Series contains no valid observations.
    """

    valid_prices = prices.dropna().sort_index()

    valid_rolling_series = (
        rolling_series
        .dropna()
        .sort_index()
    )

    if valid_prices.empty:
        raise ValueError(
            "No valid source-price observations are "
            "available for plotting."
        )

    if valid_rolling_series.empty:
        raise ValueError(
            "No valid rolling-series observations are "
            "available for plotting."
        )

    figure, axes = plt.subplots(
        figsize=figure_size
    )

    axes.plot(
        valid_prices.index,
        valid_prices.values,
        linewidth=line_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=price_label,
    )

    axes.plot(
        valid_rolling_series.index,
        valid_rolling_series.values,
        linewidth=line_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=rolling_label,
    )

    add_chart_titles(
        axes=axes,
        title=title,
        subtitle=subtitle,
    )

    axes.set_xlabel(
        x_axis_label,
        fontsize=11,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.set_ylabel(
        y_axis_label,
        fontsize=11,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.yaxis.set_major_formatter(
        FuncFormatter(format_currency_axis)
    )

    format_date_axis(
        axes=axes,
        dates=valid_prices.index,
    )

    apply_chart_layout(
        axes=axes,
        show_grid=show_grid,
    )

    if show_latest_values:
        add_latest_value_annotation(
            axes=axes,
            prices=valid_prices,
        )

        add_latest_value_annotation(
            axes=axes,
            prices=valid_rolling_series,
        )

    if show_legend:
        axes.legend(
            loc="upper left",
            frameon=False,
            fontsize=10,
        )

    figure.tight_layout(
        pad=1.5
    )

    return figure, axes

def plot_rolling_average(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    title: str = DEFAULT_ROLLING_AVERAGE_TITLE,
    subtitle: str | None = None,
    price_label: str = DEFAULT_PRICE_LABEL,
    rolling_label: str | None = None,
    x_axis_label: str = DEFAULT_X_AXIS_LABEL,
    y_axis_label: str = DEFAULT_ROLLING_Y_AXIS_LABEL,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    show_grid: bool = True,
    show_latest_values: bool = True,
    show_legend: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot gold prices alongside a rolling arithmetic average.

    The rolling average is calculated by the analytics layer
    using the requested number of monthly observations.

    Args:
        data:
            DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly observations included in each
            rolling-average calculation.

        title:
            Main chart title.

        subtitle:
            Optional subtitle. When omitted, the chart uses a
            generated date-range subtitle.

        price_label:
            Legend label for the source-price line.

        rolling_label:
            Optional legend label for the rolling-average
            line. When omitted, a label is generated from the
            selected window.

        x_axis_label:
            Label displayed beneath the horizontal axis.

        y_axis_label:
            Label displayed beside the vertical axis.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of the plotted lines.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_values:
            Whether to annotate the latest source price and
            rolling-average value.

        show_legend:
            Whether to display the chart legend.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If an argument has an invalid type.

        ValueError:
            If the requested rolling calculation cannot be
            produced.

        KeyError:
            If the requested price column is unavailable.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    normalized_column = validate_text_option(
        value=column,
        option_name="column",
    )

    normalized_title = validate_text_option(
        value=title,
        option_name="title",
    )

    normalized_subtitle = validate_optional_text(
        value=subtitle,
        option_name="subtitle",
    )

    normalized_price_label = validate_text_option(
        value=price_label,
        option_name="price_label",
    )

    normalized_rolling_label = validate_optional_text(
        value=rolling_label,
        option_name="rolling_label",
    )

    normalized_x_label = validate_text_option(
        value=x_axis_label,
        option_name="x_axis_label",
    )

    normalized_y_label = validate_text_option(
        value=y_axis_label,
        option_name="y_axis_label",
    )

    normalized_figure_size = validate_figure_size(
        figure_size=figure_size
    )

    normalized_line_width = validate_line_width(
        line_width=line_width
    )

    normalized_show_grid = validate_boolean_option(
        value=show_grid,
        option_name="show_grid",
    )

    normalized_show_latest_values = (
        validate_boolean_option(
            value=show_latest_values,
            option_name="show_latest_values",
        )
    )

    normalized_show_legend = validate_boolean_option(
        value=show_legend,
        option_name="show_legend",
    )

    prices = prepare_price_series(
        data=data,
        column=normalized_column,
    )

    rolling_average = calculate_rolling_average(
        data=data,
        column=normalized_column,
        window=window,
    )

    if normalized_subtitle is None:
        date_range = create_date_range_subtitle(
            prices=prices
        )

        normalized_subtitle = (
            f"{window}-month rolling average | "
            f"{date_range}"
        )

    if normalized_rolling_label is None:
        normalized_rolling_label = (
            DEFAULT_ROLLING_AVERAGE_LABEL.format(
                window=window
            )
        )

    return _plot_rolling_series(
        prices=prices,
        rolling_series=rolling_average,
        title=normalized_title,
        subtitle=normalized_subtitle,
        price_label=normalized_price_label,
        rolling_label=normalized_rolling_label,
        x_axis_label=normalized_x_label,
        y_axis_label=normalized_y_label,
        figure_size=normalized_figure_size,
        line_width=normalized_line_width,
        show_grid=normalized_show_grid,
        show_latest_values=(
            normalized_show_latest_values
        ),
        show_legend=normalized_show_legend,
    )

def format_percent_axis(
    value: float,
    position: int,
) -> str:
    """
    Format a decimal axis value as a percentage.

    Args:
        value:
            Decimal value supplied by Matplotlib.

        position:
            Tick position supplied by Matplotlib. This argument
            is required by FuncFormatter but is not otherwise
            used.

    Returns:
        Percentage-formatted tick label.
    """

    return f"{value * 100:,.0f}%"

def add_latest_percent_annotation(
    axes: Axes,
    values: pd.Series,
    decimal_places: int = 2,
) -> None:
    """
    Annotate the latest valid percentage value in a Series.

    Args:
        axes:
            Matplotlib Axes receiving the annotation.

        values:
            Decimal-valued Series indexed by date.

        decimal_places:
            Number of percentage decimal places to display.

    Raises:
        ValueError:
            If the Series contains no valid observations.
    """

    valid_values = values.dropna().sort_index()

    if valid_values.empty:
        raise ValueError(
            "No valid percentage values are available "
            "for annotation."
        )

    latest_date = valid_values.index[-1]
    latest_value = float(valid_values.iloc[-1])

    annotation_text = (
        f"{latest_value * 100:,.{decimal_places}f}%"
    )

    axes.annotate(
        annotation_text,
        xy=(
            latest_date,
            latest_value,
        ),
        xytext=(8, 0),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=9,
        fontweight="bold",
    )

def _plot_percentage_series(
    values: pd.Series,
    title: str,
    subtitle: str | None,
    series_label: str,
    x_axis_label: str,
    y_axis_label: str,
    figure_size: tuple[float, float],
    line_width: float,
    show_grid: bool,
    show_latest_value: bool,
    show_legend: bool,
    show_zero_line: bool,
    reference_value: float | None = None,
    reference_label: str | None = None,
) -> tuple[Figure, Axes]:
    """
    Plot a date-indexed Series containing decimal percentages.

    This private helper provides shared presentation logic for
    return, volatility, and drawdown charts.

    Args:
        values:
            Decimal-valued Series indexed by date.

        title:
            Main chart title.

        subtitle:
            Optional subtitle.

        series_label:
            Legend label for the primary line.

        x_axis_label:
            Horizontal-axis label.

        y_axis_label:
            Vertical-axis label.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of the primary line.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_value:
            Whether to annotate the latest valid value.

        show_legend:
            Whether to display the chart legend.

        show_zero_line:
            Whether to draw a horizontal line at zero.

        reference_value:
            Optional decimal value displayed as a horizontal
            reference line.

        reference_label:
            Optional legend label for the reference line.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        ValueError:
            If the Series contains no valid observations.
    """

    valid_values = values.dropna().sort_index()

    if valid_values.empty:
        raise ValueError(
            "No valid percentage observations are available "
            "for plotting."
        )

    figure, axes = plt.subplots(
        figsize=figure_size
    )

    axes.plot(
        valid_values.index,
        valid_values.values,
        linewidth=line_width,
        solid_capstyle="round",
        solid_joinstyle="round",
        label=series_label,
    )

    if show_zero_line:
        axes.axhline(
            y=0.0,
            linewidth=DEFAULT_ZERO_LINE_WIDTH,
            linestyle="--",
        )

    if reference_value is not None:
        axes.axhline(
            y=reference_value,
            linewidth=DEFAULT_REFERENCE_LINE_WIDTH,
            linestyle=":",
            label=reference_label,
        )

    add_chart_titles(
        axes=axes,
        title=title,
        subtitle=subtitle,
    )

    axes.set_xlabel(
        x_axis_label,
        fontsize=11,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.set_ylabel(
        y_axis_label,
        fontsize=11,
        labelpad=DEFAULT_LABEL_PADDING,
    )

    axes.yaxis.set_major_formatter(
        FuncFormatter(format_percent_axis)
    )

    format_date_axis(
        axes=axes,
        dates=valid_values.index,
    )

    apply_chart_layout(
        axes=axes,
        show_grid=show_grid,
    )

    if show_latest_value:
        add_latest_percent_annotation(
            axes=axes,
            values=valid_values,
        )

    if show_legend:
        axes.legend(
            loc="upper left",
            frameon=False,
            fontsize=10,
        )

    figure.tight_layout(
        pad=1.5
    )

    return figure, axes

def plot_monthly_returns(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    title: str = DEFAULT_MONTHLY_RETURNS_TITLE,
    subtitle: str | None = None,
    series_label: str = DEFAULT_MONTHLY_RETURNS_LABEL,
    x_axis_label: str = DEFAULT_X_AXIS_LABEL,
    y_axis_label: str = DEFAULT_RETURNS_Y_AXIS_LABEL,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    show_grid: bool = True,
    show_latest_value: bool = True,
    show_legend: bool = True,
    show_zero_line: bool = True,
    show_average_line: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot month-over-month percentage returns.

    Monthly returns are calculated by the analytics layer and
    represented internally as decimal values.

    Args:
        data:
            DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        title:
            Main chart title.

        subtitle:
            Optional subtitle. When omitted, a date-range
            subtitle is generated automatically.

        series_label:
            Legend label for the monthly-return line.

        x_axis_label:
            Label displayed beneath the horizontal axis.

        y_axis_label:
            Label displayed beside the vertical axis.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of the monthly-return line.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_value:
            Whether to annotate the latest monthly return.

        show_legend:
            Whether to display the chart legend.

        show_zero_line:
            Whether to display a horizontal zero-return line.

        show_average_line:
            Whether to display the arithmetic average monthly
            return as a reference line.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If an argument has an invalid type.

        ValueError:
            If monthly returns cannot be calculated.

        KeyError:
            If the requested price column is unavailable.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    normalized_column = validate_text_option(
        value=column,
        option_name="column",
    )

    normalized_title = validate_text_option(
        value=title,
        option_name="title",
    )

    normalized_subtitle = validate_optional_text(
        value=subtitle,
        option_name="subtitle",
    )

    normalized_series_label = validate_text_option(
        value=series_label,
        option_name="series_label",
    )

    normalized_x_label = validate_text_option(
        value=x_axis_label,
        option_name="x_axis_label",
    )

    normalized_y_label = validate_text_option(
        value=y_axis_label,
        option_name="y_axis_label",
    )

    normalized_figure_size = validate_figure_size(
        figure_size=figure_size
    )

    normalized_line_width = validate_line_width(
        line_width=line_width
    )

    normalized_show_grid = validate_boolean_option(
        value=show_grid,
        option_name="show_grid",
    )

    normalized_show_latest_value = (
        validate_boolean_option(
            value=show_latest_value,
            option_name="show_latest_value",
        )
    )

    normalized_show_legend = validate_boolean_option(
        value=show_legend,
        option_name="show_legend",
    )

    normalized_show_zero_line = (
        validate_boolean_option(
            value=show_zero_line,
            option_name="show_zero_line",
        )
    )

    normalized_show_average_line = (
        validate_boolean_option(
            value=show_average_line,
            option_name="show_average_line",
        )
    )

    monthly_returns = calculate_monthly_returns(
        data=data,
        column=normalized_column,
    )

    if monthly_returns.empty:
        raise ValueError(
            "No monthly returns are available for plotting."
        )

    if normalized_subtitle is None:
        normalized_subtitle = (
            create_date_range_subtitle(
                prices=monthly_returns
            )
        )

    reference_value = None
    reference_label = None

    if normalized_show_average_line:
        reference_value = float(
            monthly_returns.mean()
        )

        reference_label = (
            "Average Monthly Return "
            f"({reference_value * 100:,.2f}%)"
        )

    return _plot_percentage_series(
        values=monthly_returns,
        title=normalized_title,
        subtitle=normalized_subtitle,
        series_label=normalized_series_label,
        x_axis_label=normalized_x_label,
        y_axis_label=normalized_y_label,
        figure_size=normalized_figure_size,
        line_width=normalized_line_width,
        show_grid=normalized_show_grid,
        show_latest_value=(
            normalized_show_latest_value
        ),
        show_legend=normalized_show_legend,
        show_zero_line=normalized_show_zero_line,
        reference_value=reference_value,
        reference_label=reference_label,
    )

def plot_rolling_volatility(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    annualize: bool = True,
    title: str = DEFAULT_ROLLING_VOLATILITY_TITLE,
    subtitle: str | None = None,
    series_label: str | None = None,
    x_axis_label: str = DEFAULT_X_AXIS_LABEL,
    y_axis_label: str | None = None,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    show_grid: bool = True,
    show_latest_value: bool = True,
    show_legend: bool = True,
    show_average_line: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot rolling volatility calculated from monthly returns.

    Volatility is represented as a decimal percentage. When
    annualization is enabled, monthly volatility is converted
    into annualized volatility by the analytics layer.

    Args:
        data:
            DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly returns included in each rolling
            volatility calculation.

        annualize:
            Whether to convert monthly volatility into
            annualized volatility.

        title:
            Main chart title.

        subtitle:
            Optional subtitle. When omitted, the chart creates
            one from the rolling window and date range.

        series_label:
            Optional legend label for the volatility line.
            When omitted, a label is generated automatically.

        x_axis_label:
            Label displayed beneath the horizontal axis.

        y_axis_label:
            Optional vertical-axis label. When omitted, the
            label reflects whether volatility is annualized.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of the volatility line.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_value:
            Whether to annotate the latest volatility value.

        show_legend:
            Whether to display the chart legend.

        show_average_line:
            Whether to display average rolling volatility as
            a horizontal reference line.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If an argument has an invalid type.

        ValueError:
            If rolling volatility cannot be calculated.

        KeyError:
            If the requested price column is unavailable.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    normalized_column = validate_text_option(
        value=column,
        option_name="column",
    )

    normalized_title = validate_text_option(
        value=title,
        option_name="title",
    )

    normalized_subtitle = validate_optional_text(
        value=subtitle,
        option_name="subtitle",
    )

    normalized_series_label = validate_optional_text(
        value=series_label,
        option_name="series_label",
    )

    normalized_x_label = validate_text_option(
        value=x_axis_label,
        option_name="x_axis_label",
    )

    normalized_y_label = validate_optional_text(
        value=y_axis_label,
        option_name="y_axis_label",
    )

    normalized_figure_size = validate_figure_size(
        figure_size=figure_size
    )

    normalized_line_width = validate_line_width(
        line_width=line_width
    )

    normalized_annualize = validate_boolean_option(
        value=annualize,
        option_name="annualize",
    )

    normalized_show_grid = validate_boolean_option(
        value=show_grid,
        option_name="show_grid",
    )

    normalized_show_latest_value = (
        validate_boolean_option(
            value=show_latest_value,
            option_name="show_latest_value",
        )
    )

    normalized_show_legend = validate_boolean_option(
        value=show_legend,
        option_name="show_legend",
    )

    normalized_show_average_line = (
        validate_boolean_option(
            value=show_average_line,
            option_name="show_average_line",
        )
    )

    rolling_volatility = calculate_rolling_volatility(
        data=data,
        column=normalized_column,
        window=window,
        annualize=normalized_annualize,
    )

    valid_volatility = (
        rolling_volatility
        .dropna()
        .sort_index()
    )

    if valid_volatility.empty:
        raise ValueError(
            "No rolling volatility values are available "
            "for plotting."
        )

    if normalized_series_label is None:
        if normalized_annualize:
            normalized_series_label = (
                DEFAULT_ROLLING_VOLATILITY_LABEL.format(
                    window=window
                )
            )

        else:
            normalized_series_label = (
                DEFAULT_NON_ANNUALIZED_VOLATILITY_LABEL.format(
                    window=window
                )
            )

    if normalized_y_label is None:
        if normalized_annualize:
            normalized_y_label = (
                DEFAULT_VOLATILITY_Y_AXIS_LABEL
            )

        else:
            normalized_y_label = (
                DEFAULT_NON_ANNUALIZED_VOLATILITY_Y_AXIS_LABEL
            )

    if normalized_subtitle is None:
        date_range = create_date_range_subtitle(
            prices=valid_volatility
        )

        annualization_text = (
            "Annualized"
            if normalized_annualize
            else "Monthly"
        )

        normalized_subtitle = (
            f"{window}-month rolling window | "
            f"{annualization_text} volatility | "
            f"{date_range}"
        )

    reference_value = None
    reference_label = None

    if normalized_show_average_line:
        reference_value = float(
            valid_volatility.mean()
        )

        reference_label = (
            "Average Volatility "
            f"({reference_value * 100:,.2f}%)"
        )

    return _plot_percentage_series(
        values=valid_volatility,
        title=normalized_title,
        subtitle=normalized_subtitle,
        series_label=normalized_series_label,
        x_axis_label=normalized_x_label,
        y_axis_label=normalized_y_label,
        figure_size=normalized_figure_size,
        line_width=normalized_line_width,
        show_grid=normalized_show_grid,
        show_latest_value=(
            normalized_show_latest_value
        ),
        show_legend=normalized_show_legend,
        show_zero_line=False,
        reference_value=reference_value,
        reference_label=reference_label,
    )

def plot_rolling_return(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    title: str = DEFAULT_ROLLING_RETURN_TITLE,
    subtitle: str | None = None,
    series_label: str | None = None,
    x_axis_label: str = DEFAULT_X_AXIS_LABEL,
    y_axis_label: str = DEFAULT_ROLLING_RETURN_Y_AXIS_LABEL,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
    show_grid: bool = True,
    show_latest_value: bool = True,
    show_legend: bool = True,
    show_zero_line: bool = True,
    show_average_line: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot rolling percentage returns over a selected window.

    Rolling returns measure the percentage change between each
    observation and the observation located the requested
    number of months earlier.

    Args:
        data:
            DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly observations used for each
            rolling-return calculation.

        title:
            Main chart title.

        subtitle:
            Optional subtitle. When omitted, a subtitle is
            generated from the window and date range.

        series_label:
            Optional legend label for the rolling-return line.
            When omitted, a label is generated automatically.

        x_axis_label:
            Label displayed beneath the horizontal axis.

        y_axis_label:
            Label displayed beside the vertical axis.

        figure_size:
            Width and height of the figure in inches.

        line_width:
            Width of the rolling-return line.

        show_grid:
            Whether to display horizontal grid lines.

        show_latest_value:
            Whether to annotate the latest rolling return.

        show_legend:
            Whether to display the chart legend.

        show_zero_line:
            Whether to display a horizontal zero-return line.

        show_average_line:
            Whether to display average rolling return as a
            horizontal reference line.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If an argument has an invalid type.

        ValueError:
            If rolling returns cannot be calculated.

        KeyError:
            If the requested price column is unavailable.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    normalized_column = validate_text_option(
        value=column,
        option_name="column",
    )

    normalized_title = validate_text_option(
        value=title,
        option_name="title",
    )

    normalized_subtitle = validate_optional_text(
        value=subtitle,
        option_name="subtitle",
    )

    normalized_series_label = validate_optional_text(
        value=series_label,
        option_name="series_label",
    )

    normalized_x_label = validate_text_option(
        value=x_axis_label,
        option_name="x_axis_label",
    )

    normalized_y_label = validate_text_option(
        value=y_axis_label,
        option_name="y_axis_label",
    )

    normalized_figure_size = validate_figure_size(
        figure_size=figure_size
    )

    normalized_line_width = validate_line_width(
        line_width=line_width
    )

    normalized_show_grid = validate_boolean_option(
        value=show_grid,
        option_name="show_grid",
    )

    normalized_show_latest_value = (
        validate_boolean_option(
            value=show_latest_value,
            option_name="show_latest_value",
        )
    )

    normalized_show_legend = validate_boolean_option(
        value=show_legend,
        option_name="show_legend",
    )

    normalized_show_zero_line = (
        validate_boolean_option(
            value=show_zero_line,
            option_name="show_zero_line",
        )
    )

    normalized_show_average_line = (
        validate_boolean_option(
            value=show_average_line,
            option_name="show_average_line",
        )
    )

    rolling_return = calculate_rolling_return(
        data=data,
        column=normalized_column,
        window=window,
    )

    valid_returns = (
        rolling_return
        .dropna()
        .sort_index()
    )

    if valid_returns.empty:
        raise ValueError(
            "No rolling return values are available "
            "for plotting."
        )

    if normalized_series_label is None:
        normalized_series_label = (
            DEFAULT_ROLLING_RETURN_LABEL.format(
                window=window
            )
        )

    if normalized_subtitle is None:
        date_range = create_date_range_subtitle(
            prices=valid_returns
        )

        normalized_subtitle = (
            f"{window}-month rolling return | "
            f"{date_range}"
        )

    reference_value = None
    reference_label = None

    if normalized_show_average_line:
        reference_value = float(
            valid_returns.mean()
        )

        reference_label = (
            "Average Rolling Return "
            f"({reference_value * 100:,.2f}%)"
        )

    return _plot_percentage_series(
        values=valid_returns,
        title=normalized_title,
        subtitle=normalized_subtitle,
        series_label=normalized_series_label,
        x_axis_label=normalized_x_label,
        y_axis_label=normalized_y_label,
        figure_size=normalized_figure_size,
        line_width=normalized_line_width,
        show_grid=normalized_show_grid,
        show_latest_value=(
            normalized_show_latest_value
        ),
        show_legend=normalized_show_legend,
        show_zero_line=normalized_show_zero_line,
        reference_value=reference_value,
        reference_label=reference_label,
    )

def plot_rolling_high_low(
    data: pd.DataFrame,
    column: str = DEFAULT_PRICE_COLUMN,
    window: int = DEFAULT_ROLLING_WINDOW,
    title: str = DEFAULT_ROLLING_HIGH_LOW_TITLE,
    subtitle: str | None = None,
    price_label: str = DEFAULT_PRICE_SERIES_LABEL,
    high_label: str | None = None,
    low_label: str | None = None,
    x_axis_label: str = DEFAULT_X_AXIS_LABEL,
    y_axis_label: str = DEFAULT_ROLLING_HIGH_LOW_Y_AXIS_LABEL,
    figure_size: tuple[float, float] = DEFAULT_FIGURE_SIZE,
    price_line_width: float = DEFAULT_LINE_WIDTH,
    range_line_width: float = DEFAULT_LINE_WIDTH,
    show_grid: bool = True,
    show_legend: bool = True,
    show_latest_values: bool = True,
    show_range_fill: bool = True,
) -> tuple[Figure, Axes]:
    """
    Plot gold prices alongside rolling highs and rolling lows.

    The rolling high represents the highest price observed during
    each trailing window. The rolling low represents the lowest
    price observed during the same trailing window.

    Args:
        data:
            DataFrame containing dates and prices.

        column:
            Numeric price column to analyze.

        window:
            Number of monthly observations included in each
            rolling high and rolling low calculation.

        title:
            Main chart title.

        subtitle:
            Optional chart subtitle. When omitted, one is generated
            from the rolling window and available date range.

        price_label:
            Legend label for the original gold-price series.

        high_label:
            Optional legend label for the rolling-high series.

        low_label:
            Optional legend label for the rolling-low series.

        x_axis_label:
            Label displayed beneath the horizontal axis.

        y_axis_label:
            Label displayed beside the vertical axis.

        figure_size:
            Width and height of the figure in inches.

        price_line_width:
            Width of the original-price line.

        range_line_width:
            Width of the rolling-high and rolling-low lines.

        show_grid:
            Whether to display horizontal grid lines.

        show_legend:
            Whether to display the chart legend.

        show_latest_values:
            Whether to annotate the latest price, rolling high,
            and rolling low.

        show_range_fill:
            Whether to shade the area between the rolling high
            and rolling low.

    Returns:
        A tuple containing the Matplotlib Figure and Axes.

    Raises:
        TypeError:
            If an argument has an invalid type.

        ValueError:
            If no rolling values are available for plotting.

        KeyError:
            If the requested price column is unavailable.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    normalized_column = validate_text_option(
        value=column,
        option_name="column",
    )

    normalized_title = validate_text_option(
        value=title,
        option_name="title",
    )

    normalized_subtitle = validate_optional_text(
        value=subtitle,
        option_name="subtitle",
    )

    normalized_price_label = validate_text_option(
        value=price_label,
        option_name="price_label",
    )

    normalized_high_label = validate_optional_text(
        value=high_label,
        option_name="high_label",
    )

    normalized_low_label = validate_optional_text(
        value=low_label,
        option_name="low_label",
    )

    normalized_x_label = validate_text_option(
        value=x_axis_label,
        option_name="x_axis_label",
    )

    normalized_y_label = validate_text_option(
        value=y_axis_label,
        option_name="y_axis_label",
    )

    normalized_figure_size = validate_figure_size(
        figure_size=figure_size
    )

    normalized_price_line_width = validate_line_width(
        line_width=price_line_width
    )

    normalized_range_line_width = validate_line_width(
        line_width=range_line_width
    )

    normalized_show_grid = validate_boolean_option(
        value=show_grid,
        option_name="show_grid",
    )

    normalized_show_legend = validate_boolean_option(
        value=show_legend,
        option_name="show_legend",
    )

    normalized_show_latest_values = validate_boolean_option(
        value=show_latest_values,
        option_name="show_latest_values",
    )

    normalized_show_range_fill = validate_boolean_option(
        value=show_range_fill,
        option_name="show_range_fill",
    )

    prices = prepare_price_series(
        data=data,
        column=normalized_column,
    )

    rolling_high = calculate_rolling_high(
        data=data,
        column=normalized_column,
        window=window,
    )

    rolling_low = calculate_rolling_low(
        data=data,
        column=normalized_column,
        window=window,
    )

    chart_data = pd.concat(
        [
            prices.rename("price"),
            rolling_high.rename("rolling_high"),
            rolling_low.rename("rolling_low"),
        ],
        axis=1,
    ).dropna()

    chart_data = chart_data.sort_index()

    if chart_data.empty:
        raise ValueError(
            "No rolling high and low values are available "
            "for plotting."
        )

    if normalized_high_label is None:
        normalized_high_label = (
            DEFAULT_ROLLING_HIGH_LABEL.format(
                window=window
            )
        )

    if normalized_low_label is None:
        normalized_low_label = (
            DEFAULT_ROLLING_LOW_LABEL.format(
                window=window
            )
        )

    if normalized_subtitle is None:
        date_range = create_date_range_subtitle(
            prices=chart_data["price"]
        )

        normalized_subtitle = (
            f"{window}-month trailing price range | "
            f"{date_range}"
        )

    figure, axes = plt.subplots(
        figsize=normalized_figure_size
    )

    price_line = axes.plot(
        chart_data.index,
        chart_data["price"],
        linewidth=normalized_price_line_width,
        label=normalized_price_label,
    )[0]

    high_line = axes.plot(
        chart_data.index,
        chart_data["rolling_high"],
        linewidth=normalized_range_line_width,
        linestyle="--",
        label=normalized_high_label,
    )[0]

    low_line = axes.plot(
        chart_data.index,
        chart_data["rolling_low"],
        linewidth=normalized_range_line_width,
        linestyle="--",
        label=normalized_low_label,
    )[0]

    if normalized_show_range_fill:
        axes.fill_between(
            chart_data.index,
            chart_data["rolling_low"],
            chart_data["rolling_high"],
            alpha=DEFAULT_RANGE_FILL_ALPHA,
        )

    axes.set_title(
        normalized_title,
        loc="left",
    )

    axes.text(
        0.0,
        1.01,
        normalized_subtitle,
        transform=axes.transAxes,
        ha="left",
        va="bottom",
    )

    axes.set_xlabel(normalized_x_label)
    axes.set_ylabel(normalized_y_label)

    axes.yaxis.set_major_formatter(
    FuncFormatter(format_currency_axis)
)

    axes.grid(
        visible=normalized_show_grid,
        axis="y",
        alpha=DEFAULT_GRID_ALPHA,
    )

    if normalized_show_latest_values:
        latest_date = chart_data.index[-1]

        latest_price = float(
            chart_data["price"].iloc[-1]
        )

        latest_high = float(
            chart_data["rolling_high"].iloc[-1]
        )

        latest_low = float(
            chart_data["rolling_low"].iloc[-1]
        )

        axes.annotate(
            f"Price: ${latest_price:,.2f}",
            xy=(latest_date, latest_price),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=price_line.get_color(),
        )

        axes.annotate(
            f"High: ${latest_high:,.2f}",
            xy=(latest_date, latest_high),
            xytext=(10, 10),
            textcoords="offset points",
            va="bottom",
            color=high_line.get_color(),
        )

        axes.annotate(
            f"Low: ${latest_low:,.2f}",
            xy=(latest_date, latest_low),
            xytext=(10, -10),
            textcoords="offset points",
            va="top",
            color=low_line.get_color(),
        )

        axes.margins(x=0.08)

    if normalized_show_legend:
        axes.legend()

    figure.tight_layout()

    return figure, axes