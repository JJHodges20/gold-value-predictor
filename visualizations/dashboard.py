"""
Dashboard architecture and rendering for the Gold Value Predictor.

This module defines dashboard configuration, validation, metadata,
summary metrics, layout creation, panel rendering, and orchestration.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from analytics.forecasting import (
    DEFAULT_EXPECTED_GROWTH_RATE,
    DEFAULT_FORECAST_HISTORY_YEARS,
    DEFAULT_FORECAST_YEARS,
    generate_forecast_series,
    project_future_value,
    validate_growth_rate,
    validate_years,
)

from analytics.inflation import calculate_real_price

from analytics.returns import (
    DEFAULT_PRICE_COLUMN,
    calculate_annual_returns,
    calculate_cagr,
    calculate_monthly_returns,
    prepare_price_series,
)

from analytics.risk import (
    calculate_annualized_volatility,
    calculate_max_drawdown,
)
from analytics.rolling import (
    DEFAULT_ROLLING_WINDOW,
    calculate_rolling_drawdown,
    calculate_rolling_volatility,
)

from data_sources.merger import DATE_COLUMN

from visualizations.charts import (
    format_currency_axis,
    format_percentage_axis,
)
from visualizations.styling import (
    BACKGROUND_COLOR,
    DEFAULT_FONT_FAMILY,
    DEFAULT_LINE_WIDTH,
    HISTORICAL_PRICE_COLOR,
    NEGATIVE_RETURN_COLOR,
    NOMINAL_PRICE_COLOR,
    PLOT_BACKGROUND_COLOR,
    POSITIVE_RETURN_COLOR,
    REAL_PRICE_COLOR,
    RECENT_PRICE_COLOR,
    REFERENCE_LINE_COLOR,
    REFERENCE_LINE_WIDTH,
    SECONDARY_TEXT_COLOR,
    SUBTITLE_FONT_SIZE,
    TEXT_COLOR,
    TITLE_FONT_SIZE,
    TITLE_FONT_WEIGHT,
    apply_standard_formatting,
    DRAWDOWN_COLOR,
    REFERENCE_LINE_COLOR,
    REFERENCE_LINE_WIDTH,
    VOLATILITY_COLOR,
    FORECAST_BOUNDARY_COLOR,
    FORECAST_COLOR,
    annotate_latest_value,
    annotate_vertical_event,
)

# ------------------------------------------------------------------
# Dashboard rolling-panel labels
# ------------------------------------------------------------------


DASHBOARD_VOLATILITY_LABEL = (
    "Annualized Volatility"
)

DASHBOARD_DRAWDOWN_LABEL = (
    "Rolling Drawdown"
)

DASHBOARD_PERCENTAGE_LABEL = (
    "Percentage"
)

DASHBOARD_ROLLING_HIGH_LABEL = (
    "Rolling High"
)

DASHBOARD_DRAWDOWN_FILL_ALPHA = 0.18

# ------------------------------------------------------------------
# Dashboard forecast-panel settings
# ------------------------------------------------------------------


DASHBOARD_FORECAST_HISTORY_LABEL = (
    "Historical Price"
)

DASHBOARD_FORECAST_LABEL = (
    "Projected Price"
)

DASHBOARD_FORECAST_BOUNDARY_LABEL = (
    "Forecast Begins"
)

DASHBOARD_FORECAST_LINE_STYLE = "--"

DASHBOARD_FORECAST_BOUNDARY_STYLE = ":"

DASHBOARD_FORECAST_MARKER_SIZE = 5.0

DASHBOARD_FORECAST_DISCLAIMER = (
    "Forecast values are hypothetical compound-growth "
    "scenarios, not predictions or investment advice."
)

DASHBOARD_FORECAST_DISCLAIMER_X = 0.01

DASHBOARD_FORECAST_DISCLAIMER_Y = 0.02

# ------------------------------------------------------------------
# Dashboard defaults
# ------------------------------------------------------------------


DEFAULT_DASHBOARD_TITLE = (
    "Gold Value Predictor Dashboard"
)

DEFAULT_DASHBOARD_SUBTITLE = (
    "Historical performance, inflation, risk, "
    "rolling analysis, and hypothetical forecasts"
)

DEFAULT_DASHBOARD_FIGURE_SIZE = (
    18.0,
    24.0,
)

DEFAULT_DASHBOARD_DPI = 100

DEFAULT_DASHBOARD_RECENT_YEARS = 5

DEFAULT_DASHBOARD_ROLLING_WINDOW = (
    DEFAULT_ROLLING_WINDOW
)

DEFAULT_DASHBOARD_FORECAST_YEARS = (
    DEFAULT_FORECAST_YEARS
)

DEFAULT_DASHBOARD_FORECAST_HISTORY_YEARS = (
    DEFAULT_FORECAST_HISTORY_YEARS
)

DEFAULT_DASHBOARD_GROWTH_RATE = (
    DEFAULT_EXPECTED_GROWTH_RATE
)

DEFAULT_DASHBOARD_SOURCE_NOTE = (
    "Source: Gold Value Predictor master dataset"
)

DEFAULT_DASHBOARD_WATERMARK = (
    "Gold Value Predictor"
)


# ------------------------------------------------------------------
# Dashboard panel identifiers
# ------------------------------------------------------------------


PANEL_SUMMARY = "summary"

PANEL_HISTORICAL_PRICE = (
    "historical_price"
)

PANEL_RECENT_PRICE = (
    "recent_price"
)

PANEL_NOMINAL_VS_REAL = (
    "nominal_vs_real"
)

PANEL_MONTHLY_RETURNS = (
    "monthly_returns"
)

PANEL_ANNUAL_RETURNS = (
    "annual_returns"
)

PANEL_ROLLING_VOLATILITY = (
    "rolling_volatility"
)

PANEL_ROLLING_DRAWDOWN = (
    "rolling_drawdown"
)

PANEL_FORECAST = "forecast"


DASHBOARD_PANEL_ORDER = (
    PANEL_SUMMARY,
    PANEL_HISTORICAL_PRICE,
    PANEL_RECENT_PRICE,
    PANEL_NOMINAL_VS_REAL,
    PANEL_MONTHLY_RETURNS,
    PANEL_ANNUAL_RETURNS,
    PANEL_ROLLING_VOLATILITY,
    PANEL_ROLLING_DRAWDOWN,
    PANEL_FORECAST,
)


DASHBOARD_PANEL_TITLES = {
    PANEL_SUMMARY: "Summary",
    PANEL_HISTORICAL_PRICE: (
        "Historical Gold Price"
    ),
    PANEL_RECENT_PRICE: (
        "Recent Gold Price"
    ),
    PANEL_NOMINAL_VS_REAL: (
        "Nominal vs. Inflation-Adjusted Price"
    ),
    PANEL_MONTHLY_RETURNS: (
        "Monthly Returns"
    ),
    PANEL_ANNUAL_RETURNS: (
        "Annual Returns"
    ),
    PANEL_ROLLING_VOLATILITY: (
        "Rolling Volatility"
    ),
    PANEL_ROLLING_DRAWDOWN: (
        "Rolling Drawdown"
    ),
    PANEL_FORECAST: (
        "Hypothetical Forecast"
    ),
}


# ------------------------------------------------------------------
# Dashboard chart labels
# ------------------------------------------------------------------


DASHBOARD_DATE_LABEL = "Date"

DASHBOARD_PRICE_LABEL = (
    "Gold Price (USD)"
)

DASHBOARD_HISTORICAL_LABEL = (
    "Gold Price"
)

DASHBOARD_NOMINAL_LABEL = (
    "Nominal Price"
)

DASHBOARD_REAL_LABEL = (
    "Inflation-Adjusted Price"
)


# ------------------------------------------------------------------
# Dashboard layout settings
# ------------------------------------------------------------------


DASHBOARD_GRID_ROWS = 6

DASHBOARD_GRID_COLUMNS = 2

DASHBOARD_HEIGHT_RATIOS = (
    0.75,
    1.75,
    1.5,
    1.5,
    1.5,
    1.75,
)

DASHBOARD_WIDTH_RATIOS = (
    1.0,
    1.0,
)

DASHBOARD_HORIZONTAL_SPACE = 0.22

DASHBOARD_VERTICAL_SPACE = 0.42

DASHBOARD_LEFT_MARGIN = 0.07

DASHBOARD_RIGHT_MARGIN = 0.97

DASHBOARD_TOP_MARGIN = 0.91

DASHBOARD_BOTTOM_MARGIN = 0.05

DASHBOARD_TITLE_X_POSITION = 0.07

DASHBOARD_TITLE_Y_POSITION = 0.975

DASHBOARD_SUBTITLE_Y_POSITION = 0.949


# ------------------------------------------------------------------
# Dashboard summary settings
# ------------------------------------------------------------------


SUMMARY_LATEST_PRICE = (
    "latest_price"
)

SUMMARY_REAL_PRICE = (
    "inflation_adjusted_price"
)

SUMMARY_CAGR = "cagr"

SUMMARY_ANNUALIZED_VOLATILITY = (
    "annualized_volatility"
)

SUMMARY_MAX_DRAWDOWN = (
    "maximum_drawdown"
)

SUMMARY_PROJECTED_VALUE = (
    "projected_value"
)


DASHBOARD_SUMMARY_ORDER = (
    SUMMARY_LATEST_PRICE,
    SUMMARY_REAL_PRICE,
    SUMMARY_CAGR,
    SUMMARY_ANNUALIZED_VOLATILITY,
    SUMMARY_MAX_DRAWDOWN,
    SUMMARY_PROJECTED_VALUE,
)


DASHBOARD_SUMMARY_LABELS = {
    SUMMARY_LATEST_PRICE: (
        "Latest Gold Price"
    ),
    SUMMARY_REAL_PRICE: (
        "Inflation-Adjusted Price"
    ),
    SUMMARY_CAGR: (
        "Historical CAGR"
    ),
    SUMMARY_ANNUALIZED_VOLATILITY: (
        "Annualized Volatility"
    ),
    SUMMARY_MAX_DRAWDOWN: (
        "Maximum Drawdown"
    ),
    SUMMARY_PROJECTED_VALUE: (
        "Projected Future Value"
    ),
}


DASHBOARD_SUMMARY_COLUMNS = 6

DASHBOARD_SUMMARY_LABEL_Y = 0.68

DASHBOARD_SUMMARY_VALUE_Y = 0.35

DASHBOARD_SUMMARY_NOTE_Y = 0.08

# ------------------------------------------------------------------
# Dashboard returns-panel labels
# ------------------------------------------------------------------


DASHBOARD_RETURN_LABEL = "Return"

DASHBOARD_MONTHLY_RETURN_LABEL = (
    "Monthly Return"
)

DASHBOARD_POSITIVE_MONTHLY_RETURN_LABEL = (
    "Positive Monthly Return"
)

DASHBOARD_NEGATIVE_MONTHLY_RETURN_LABEL = (
    "Negative Monthly Return"
)

DASHBOARD_ANNUAL_RETURN_LABEL = (
    "Annual Return"
)

DASHBOARD_ZERO_RETURN_LABEL = (
    "Zero Return"
)

DASHBOARD_YEAR_LABEL = "Year"

# ------------------------------------------------------------------
# Dashboard data structures
# ------------------------------------------------------------------


@dataclass(
    frozen=True,
    slots=True,
)
class DashboardConfig:
    """
    Immutable configuration for dashboard generation.
    """

    title: str = DEFAULT_DASHBOARD_TITLE

    subtitle: str | None = (
        DEFAULT_DASHBOARD_SUBTITLE
    )

    price_column: str = (
        DEFAULT_PRICE_COLUMN
    )

    recent_years: int = (
        DEFAULT_DASHBOARD_RECENT_YEARS
    )

    rolling_window: int = (
        DEFAULT_DASHBOARD_ROLLING_WINDOW
    )

    forecast_years: int = (
        DEFAULT_DASHBOARD_FORECAST_YEARS
    )

    forecast_history_years: int | None = (
        DEFAULT_DASHBOARD_FORECAST_HISTORY_YEARS
    )

    annual_growth_rate: float = (
        DEFAULT_DASHBOARD_GROWTH_RATE
    )

    figure_size: tuple[float, float] = (
        DEFAULT_DASHBOARD_FIGURE_SIZE
    )

    dpi: int = DEFAULT_DASHBOARD_DPI

    source_note: str | None = (
        DEFAULT_DASHBOARD_SOURCE_NOTE
    )

    watermark: str | None = (
        DEFAULT_DASHBOARD_WATERMARK
    )

    show_annotations: bool = True

    show_forecast_disclaimer: bool = True


@dataclass(
    frozen=True,
    slots=True,
)
class DashboardContext:
    """
    Validated dashboard input and metadata.
    """

    data: pd.DataFrame

    config: DashboardConfig

    metadata: dict[str, Any]


@dataclass(
    frozen=True,
    slots=True,
)
class DashboardSummary:
    """
    Key metrics displayed in the dashboard summary panel.
    """

    latest_price: float

    inflation_adjusted_price: float

    cagr: float

    annualized_volatility: float

    maximum_drawdown: float

    projected_value: float

    latest_date: pd.Timestamp

    drawdown_peak_date: pd.Timestamp

    drawdown_trough_date: pd.Timestamp


@dataclass(
    frozen=True,
    slots=True,
)
class DashboardResult:
    """
    Completed dashboard output.
    """

    figure: Figure

    axes: dict[str, Axes]

    config: DashboardConfig

    metadata: dict[str, Any]

# ------------------------------------------------------------------
# Display formatting helpers
# ------------------------------------------------------------------


def format_dashboard_currency(
    value: Real,
) -> str:
    """
    Format a dashboard value as U.S. currency.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise TypeError(
            "value must be numeric."
        )

    return f"${float(value):,.2f}"


def format_dashboard_percentage(
    value: Real,
) -> str:
    """
    Format a decimal dashboard value as a percentage.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
    ):
        raise TypeError(
            "value must be numeric."
        )

    return f"{float(value):.2%}"


def format_dashboard_date(
    value: pd.Timestamp,
) -> str:
    """
    Format a dashboard date for display.
    """

    if not isinstance(
        value,
        pd.Timestamp,
    ):
        raise TypeError(
            "value must be a Pandas Timestamp."
        )

    return value.strftime(
        "%b %Y"
    )


def format_dashboard_date_axis(
    axes: Axes,
) -> Axes:
    """
    Apply concise date formatting to a dashboard panel.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    locator = mdates.AutoDateLocator()

    axes.xaxis.set_major_locator(
        locator
    )

    axes.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(
            locator
        )
    )

    return axes


# ------------------------------------------------------------------
# Basic validation helpers
# ------------------------------------------------------------------


def validate_dashboard_text(
    value: str | None,
    parameter_name: str,
    *,
    allow_none: bool = True,
) -> str | None:
    """
    Validate dashboard text configuration.
    """

    if value is None:
        if allow_none:
            return None

        raise TypeError(
            f"{parameter_name} must be a string."
        )

    if not isinstance(
        value,
        str,
    ):
        expected_type = (
            "a string or None"
            if allow_none
            else "a string"
        )

        raise TypeError(
            f"{parameter_name} must be "
            f"{expected_type}."
        )

    if not value.strip():
        raise ValueError(
            f"{parameter_name} cannot be empty."
        )

    return value


def validate_positive_integer(
    value: int,
    parameter_name: str,
) -> int:
    """
    Validate a positive integer dashboard option.
    """

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise TypeError(
            f"{parameter_name} must be an integer."
        )

    if value < 1:
        raise ValueError(
            f"{parameter_name} must be at least 1."
        )

    return value


def validate_dashboard_figure_size(
    figure_size: tuple[float, float],
) -> tuple[float, float]:
    """
    Validate dashboard Figure dimensions.
    """

    if (
        not isinstance(figure_size, tuple)
        or len(figure_size) != 2
    ):
        raise TypeError(
            "figure_size must be a two-value tuple."
        )

    width, height = figure_size

    for value in (
        width,
        height,
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
        ):
            raise TypeError(
                "figure_size values must be numeric."
            )

        if value <= 0:
            raise ValueError(
                "figure_size values must be greater than zero."
            )

    return (
        float(width),
        float(height),
    )


def validate_dashboard_boolean(
    value: bool,
    parameter_name: str,
) -> bool:
    """
    Validate a Boolean dashboard option.
    """

    if not isinstance(
        value,
        bool,
    ):
        raise TypeError(
            f"{parameter_name} must be a Boolean value."
        )

    return value


def validate_dashboard_config(
    config: DashboardConfig,
) -> DashboardConfig:
    """
    Validate a complete DashboardConfig.
    """

    if not isinstance(
        config,
        DashboardConfig,
    ):
        raise TypeError(
            "config must be a DashboardConfig."
        )

    validate_dashboard_text(
        config.title,
        "title",
        allow_none=False,
    )

    validate_dashboard_text(
        config.subtitle,
        "subtitle",
    )

    validate_dashboard_text(
        config.price_column,
        "price_column",
        allow_none=False,
    )

    validate_positive_integer(
        config.recent_years,
        "recent_years",
    )

    validate_positive_integer(
        config.rolling_window,
        "rolling_window",
    )

    validate_years(
        config.forecast_years,
        "forecast_years",
    )

    if (
        config.forecast_history_years
        is not None
    ):
        validate_years(
            config.forecast_history_years,
            "forecast_history_years",
        )

    validate_growth_rate(
        config.annual_growth_rate,
        "annual_growth_rate",
    )

    validate_dashboard_figure_size(
        config.figure_size
    )

    validate_positive_integer(
        config.dpi,
        "dpi",
    )

    validate_dashboard_text(
        config.source_note,
        "source_note",
    )

    validate_dashboard_text(
        config.watermark,
        "watermark",
    )

    validate_dashboard_boolean(
        config.show_annotations,
        "show_annotations",
    )

    validate_dashboard_boolean(
        config.show_forecast_disclaimer,
        "show_forecast_disclaimer",
    )

    return config


def validate_dashboard_data(
    data: pd.DataFrame,
    *,
    price_column: str = DEFAULT_PRICE_COLUMN,
) -> pd.DataFrame:
    """
    Validate dashboard source data.
    """

    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(
            "data must be a Pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Dashboard data cannot be empty."
        )

    required_columns = {
        DATE_COLUMN,
        price_column,
    }

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        missing_text = ", ".join(
            sorted(missing_columns)
        )

        raise ValueError(
            "Dashboard data is missing required "
            f"columns: {missing_text}."
        )

    validated_data = data.copy()

    validated_data[
        DATE_COLUMN
    ] = pd.to_datetime(
        validated_data[DATE_COLUMN],
        errors="coerce",
    )

    if validated_data[
        DATE_COLUMN
    ].isna().any():
        raise ValueError(
            "Dashboard data contains invalid Date values."
        )

    validated_data[
        price_column
    ] = pd.to_numeric(
        validated_data[price_column],
        errors="coerce",
    )

    if validated_data[
        price_column
    ].dropna().empty:
        raise ValueError(
            "Dashboard data contains no usable "
            f"values in {price_column}."
        )

    return (
        validated_data
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

# ------------------------------------------------------------------
# Metadata and context
# ------------------------------------------------------------------


def build_dashboard_metadata(
    data: pd.DataFrame,
    config: DashboardConfig,
) -> dict[str, Any]:
    """
    Build metadata used by dashboard panels and exports.
    """

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    price_values = validated_data[
        validated_config.price_column
    ].dropna()

    latest_price_index = (
        price_values.index[-1]
    )

    latest_price = float(
        price_values.loc[
            latest_price_index
        ]
    )

    latest_date = validated_data.loc[
        latest_price_index,
        DATE_COLUMN,
    ]

    return {
        "title": validated_config.title,
        "subtitle": validated_config.subtitle,
        "row_count": int(
            len(validated_data)
        ),
        "column_count": int(
            len(validated_data.columns)
        ),
        "date_start": validated_data[
            DATE_COLUMN
        ].min(),
        "date_end": validated_data[
            DATE_COLUMN
        ].max(),
        "latest_date": latest_date,
        "latest_price": latest_price,
        "price_column": (
            validated_config.price_column
        ),
        "recent_years": (
            validated_config.recent_years
        ),
        "rolling_window": (
            validated_config.rolling_window
        ),
        "forecast_years": (
            validated_config.forecast_years
        ),
        "forecast_history_years": (
            validated_config.forecast_history_years
        ),
        "annual_growth_rate": (
            validated_config.annual_growth_rate
        ),
        "panel_count": len(
            DASHBOARD_PANEL_ORDER
        ),
        "panel_order": (
            DASHBOARD_PANEL_ORDER
        ),
    }


def create_dashboard_context(
    data: pd.DataFrame,
    config: DashboardConfig | None = None,
) -> DashboardContext:
    """
    Create validated dashboard input and metadata.
    """

    resolved_config = (
        DashboardConfig()
        if config is None
        else config
    )

    validated_config = (
        validate_dashboard_config(
            resolved_config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    metadata = build_dashboard_metadata(
        validated_data,
        validated_config,
    )

    return DashboardContext(
        data=validated_data,
        config=validated_config,
        metadata=metadata,
    )

def build_dashboard_summary(
    data: pd.DataFrame,
    config: DashboardConfig,
) -> DashboardSummary:
    """
    Calculate the key metrics shown in the summary panel.

    Args:
        data:
            Validated dashboard source data.

        config:
            Validated dashboard configuration.

    Returns:
        DashboardSummary containing the dashboard's headline
        analytics.
    """

    validated_config = validate_dashboard_config(
        config
    )

    validated_data = validate_dashboard_data(
        data,
        price_column=(
            validated_config.price_column
        ),
    )

    prices = prepare_price_series(
        data=validated_data,
        column=(
            validated_config.price_column
        ),
    )

    if prices.empty:
        raise ValueError(
            "No valid prices are available for "
            "the dashboard summary."
        )

    latest_date = pd.Timestamp(
        prices.index[-1]
    )

    latest_price = float(
        prices.iloc[-1]
    )

    real_prices = calculate_real_price(
        data=validated_data,
        price_column=(
            validated_config.price_column
        ),
    ).dropna()

    if real_prices.empty:
        raise ValueError(
            "No inflation-adjusted prices are "
            "available for the dashboard summary."
        )

    historical_cagr = calculate_cagr(
        data=validated_data,
        column=(
            validated_config.price_column
        ),
    )

    annualized_volatility = (
        calculate_annualized_volatility(
            data=validated_data,
            column=(
                validated_config.price_column
            ),
        )
    )

    drawdown_summary = calculate_max_drawdown(
        data=validated_data,
        column=(
            validated_config.price_column
        ),
    )

    projected_value = project_future_value(
        current_price=latest_price,
        annual_growth_rate=(
            validated_config.annual_growth_rate
        ),
        years=(
            validated_config.forecast_years
        ),
    )

    return DashboardSummary(
        latest_price=latest_price,
        inflation_adjusted_price=float(
            real_prices.iloc[-1]
        ),
        cagr=float(
            historical_cagr
        ),
        annualized_volatility=float(
            annualized_volatility
        ),
        maximum_drawdown=float(
            drawdown_summary["drawdown"]
        ),
        projected_value=float(
            projected_value
        ),
        latest_date=latest_date,
        drawdown_peak_date=pd.Timestamp(
            drawdown_summary["peak_date"]
        ),
        drawdown_trough_date=pd.Timestamp(
            drawdown_summary["trough_date"]
        ),
    )


# ------------------------------------------------------------------
# Dashboard summary calculations
# ------------------------------------------------------------------


def format_dashboard_summary_values(
    summary: DashboardSummary,
) -> dict[str, str]:
    """
    Convert dashboard summary metrics into display strings.
    """

    if not isinstance(
        summary,
        DashboardSummary,
    ):
        raise TypeError(
            "summary must be a DashboardSummary."
        )

    return {
        SUMMARY_LATEST_PRICE: (
            format_dashboard_currency(
                summary.latest_price
            )
        ),
        SUMMARY_REAL_PRICE: (
            format_dashboard_currency(
                summary.inflation_adjusted_price
            )
        ),
        SUMMARY_CAGR: (
            format_dashboard_percentage(
                summary.cagr
            )
        ),
        SUMMARY_ANNUALIZED_VOLATILITY: (
            format_dashboard_percentage(
                summary.annualized_volatility
            )
        ),
        SUMMARY_MAX_DRAWDOWN: (
            format_dashboard_percentage(
                summary.maximum_drawdown
            )
        ),
        SUMMARY_PROJECTED_VALUE: (
            format_dashboard_currency(
                summary.projected_value
            )
        ),
    }

# ------------------------------------------------------------------
# Dashboard layout creation
# ------------------------------------------------------------------


def create_dashboard_figure(
    config: DashboardConfig,
) -> Figure:
    """
    Create the main dashboard Figure.
    """

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    return plt.figure(
        figsize=(
            validated_config.figure_size
        ),
        dpi=validated_config.dpi,
        facecolor=BACKGROUND_COLOR,
    )


def create_dashboard_grid(
    figure: Figure,
) -> GridSpec:
    """
    Create the dashboard GridSpec.
    """

    if not isinstance(
        figure,
        Figure,
    ):
        raise TypeError(
            "figure must be a matplotlib Figure."
        )

    return figure.add_gridspec(
        nrows=DASHBOARD_GRID_ROWS,
        ncols=DASHBOARD_GRID_COLUMNS,
        height_ratios=(
            DASHBOARD_HEIGHT_RATIOS
        ),
        width_ratios=(
            DASHBOARD_WIDTH_RATIOS
        ),
        hspace=DASHBOARD_VERTICAL_SPACE,
        wspace=DASHBOARD_HORIZONTAL_SPACE,
        left=DASHBOARD_LEFT_MARGIN,
        right=DASHBOARD_RIGHT_MARGIN,
        top=DASHBOARD_TOP_MARGIN,
        bottom=DASHBOARD_BOTTOM_MARGIN,
    )


def create_dashboard_axes(
    figure: Figure,
    grid: GridSpec,
) -> dict[str, Axes]:
    """
    Create every named dashboard panel.
    """

    if not isinstance(
        figure,
        Figure,
    ):
        raise TypeError(
            "figure must be a matplotlib Figure."
        )

    if not isinstance(
        grid,
        GridSpec,
    ):
        raise TypeError(
            "grid must be a matplotlib GridSpec."
        )

    return {
        PANEL_SUMMARY: figure.add_subplot(
            grid[0, :]
        ),
        PANEL_HISTORICAL_PRICE: (
            figure.add_subplot(
                grid[1, :]
            )
        ),
        PANEL_RECENT_PRICE: (
            figure.add_subplot(
                grid[2, 0]
            )
        ),
        PANEL_NOMINAL_VS_REAL: (
            figure.add_subplot(
                grid[2, 1]
            )
        ),
        PANEL_MONTHLY_RETURNS: (
            figure.add_subplot(
                grid[3, 0]
            )
        ),
        PANEL_ANNUAL_RETURNS: (
            figure.add_subplot(
                grid[3, 1]
            )
        ),
        PANEL_ROLLING_VOLATILITY: (
            figure.add_subplot(
                grid[4, 0]
            )
        ),
        PANEL_ROLLING_DRAWDOWN: (
            figure.add_subplot(
                grid[4, 1]
            )
        ),
        PANEL_FORECAST: (
            figure.add_subplot(
                grid[5, :]
            )
        ),
    }


def add_dashboard_heading(
    figure: Figure,
    config: DashboardConfig,
) -> None:
    """
    Add the dashboard title and optional subtitle.
    """

    if not isinstance(
        figure,
        Figure,
    ):
        raise TypeError(
            "figure must be a matplotlib Figure."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    figure.text(
        DASHBOARD_TITLE_X_POSITION,
        DASHBOARD_TITLE_Y_POSITION,
        validated_config.title,
        fontsize=TITLE_FONT_SIZE + 4,
        fontweight=TITLE_FONT_WEIGHT,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=TEXT_COLOR,
        horizontalalignment="left",
        verticalalignment="top",
    )

    if validated_config.subtitle is not None:
        figure.text(
            DASHBOARD_TITLE_X_POSITION,
            DASHBOARD_SUBTITLE_Y_POSITION,
            validated_config.subtitle,
            fontsize=SUBTITLE_FONT_SIZE,
            fontfamily=DEFAULT_FONT_FAMILY,
            color=SECONDARY_TEXT_COLOR,
            horizontalalignment="left",
            verticalalignment="top",
        )


def style_dashboard_panel(
    axes: Axes,
    panel_name: str,
) -> Axes:
    """
    Apply baseline styling to one dashboard panel.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    if not isinstance(
        panel_name,
        str,
    ):
        raise TypeError(
            "panel_name must be a string."
        )

    if (
        panel_name
        not in DASHBOARD_PANEL_TITLES
    ):
        raise ValueError(
            f"Unknown dashboard panel: "
            f"{panel_name!r}."
        )

    axes.set_facecolor(
        PLOT_BACKGROUND_COLOR
    )

    if panel_name == PANEL_SUMMARY:
        axes.set_title(
            DASHBOARD_PANEL_TITLES[
                panel_name
            ],
            loc="left",
            fontsize=TITLE_FONT_SIZE,
            fontweight=TITLE_FONT_WEIGHT,
            fontfamily=DEFAULT_FONT_FAMILY,
            color=TEXT_COLOR,
            pad=12,
        )

        axes.set_xticks([])
        axes.set_yticks([])

        for spine in axes.spines.values():
            spine.set_visible(False)

        return axes

    apply_standard_formatting(
        axes.figure,
        axes,
        title=(
            DASHBOARD_PANEL_TITLES[
                panel_name
            ]
        ),
        show_grid=True,
        show_legend=False,
        apply_tight_layout=False,
    )

    return axes


def style_dashboard_axes(
    axes: dict[str, Axes],
) -> dict[str, Axes]:
    """
    Apply baseline styling to all dashboard panels.
    """

    if not isinstance(
        axes,
        dict,
    ):
        raise TypeError(
            "axes must be a dictionary."
        )

    missing_panels = (
        set(DASHBOARD_PANEL_ORDER)
        - set(axes)
    )

    if missing_panels:
        missing_text = ", ".join(
            sorted(missing_panels)
        )

        raise ValueError(
            "Dashboard axes are missing panels: "
            f"{missing_text}."
        )

    for panel_name in (
        DASHBOARD_PANEL_ORDER
    ):
        style_dashboard_panel(
            axes[panel_name],
            panel_name,
        )

    return axes

# ------------------------------------------------------------------
# Summary-panel rendering
# ------------------------------------------------------------------


def populate_summary_panel(
    axes: Axes,
    summary: DashboardSummary,
    config: DashboardConfig,
) -> Axes:
    """
    Populate the dashboard summary panel with key metrics.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    if not isinstance(
        summary,
        DashboardSummary,
    ):
        raise TypeError(
            "summary must be a DashboardSummary."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    formatted_values = (
        format_dashboard_summary_values(
            summary
        )
    )

    axes.clear()

    axes.set_facecolor(
        PLOT_BACKGROUND_COLOR
    )

    axes.set_title(
        DASHBOARD_PANEL_TITLES[
            PANEL_SUMMARY
        ],
        loc="left",
        fontsize=TITLE_FONT_SIZE,
        fontweight=TITLE_FONT_WEIGHT,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=TEXT_COLOR,
        pad=12,
    )

    axes.set_xticks([])
    axes.set_yticks([])

    for spine in axes.spines.values():
        spine.set_visible(False)

    column_width = (
        1.0
        / DASHBOARD_SUMMARY_COLUMNS
    )

    for metric_index, metric_name in enumerate(
        DASHBOARD_SUMMARY_ORDER
    ):
        x_position = (
            metric_index * column_width
            + column_width / 2
        )

        axes.text(
            x_position,
            DASHBOARD_SUMMARY_LABEL_Y,
            DASHBOARD_SUMMARY_LABELS[
                metric_name
            ],
            transform=axes.transAxes,
            fontsize=SUBTITLE_FONT_SIZE,
            fontfamily=DEFAULT_FONT_FAMILY,
            color=SECONDARY_TEXT_COLOR,
            horizontalalignment="center",
            verticalalignment="center",
        )

        axes.text(
            x_position,
            DASHBOARD_SUMMARY_VALUE_Y,
            formatted_values[
                metric_name
            ],
            transform=axes.transAxes,
            fontsize=TITLE_FONT_SIZE,
            fontweight=TITLE_FONT_WEIGHT,
            fontfamily=DEFAULT_FONT_FAMILY,
            color=TEXT_COLOR,
            horizontalalignment="center",
            verticalalignment="center",
        )

    axes.text(
        0.01,
        DASHBOARD_SUMMARY_NOTE_Y,
        (
            f"Latest observation: "
            f"{format_dashboard_date(summary.latest_date)}"
            f"  |  Forecast assumption: "
            f"{validated_config.annual_growth_rate:.1%} "
            f"annual growth for "
            f"{validated_config.forecast_years} years"
        ),
        transform=axes.transAxes,
        fontsize=SUBTITLE_FONT_SIZE,
        fontfamily=DEFAULT_FONT_FAMILY,
        color=SECONDARY_TEXT_COLOR,
        horizontalalignment="left",
        verticalalignment="bottom",
    )

    return axes

# ------------------------------------------------------------------
# Price and inflation panels
# ------------------------------------------------------------------


def populate_historical_panel(
    axes: Axes,
    data: pd.DataFrame,
    config: DashboardConfig,
) -> Axes:
    """
    Populate the complete historical-price panel.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    prices = prepare_price_series(
        data=validated_data,
        column=(
            validated_config.price_column
        ),
    )

    axes.clear()

    axes.plot(
        prices.index,
        prices.values,
        color=HISTORICAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DASHBOARD_HISTORICAL_LABEL,
    )

    format_dashboard_date_axis(
        axes
    )

    apply_standard_formatting(
        axes.figure,
        axes,
        title=(
            DASHBOARD_PANEL_TITLES[
                PANEL_HISTORICAL_PRICE
            ]
        ),
        x_label=DASHBOARD_DATE_LABEL,
        y_label=DASHBOARD_PRICE_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_currency_axis(
        axes
    )

    return axes


def populate_recent_panel(
    axes: Axes,
    data: pd.DataFrame,
    config: DashboardConfig,
) -> Axes:
    """
    Populate the recent-price dashboard panel.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    prices = prepare_price_series(
        data=validated_data,
        column=(
            validated_config.price_column
        ),
    )

    latest_date = prices.index.max()

    start_date = (
        latest_date
        - pd.DateOffset(
            years=(
                validated_config.recent_years
            )
        )
    )

    recent_prices = prices.loc[
        prices.index >= start_date
    ]

    if recent_prices.empty:
        raise ValueError(
            "No recent prices are available "
            "for the dashboard panel."
        )

    axes.clear()

    axes.plot(
        recent_prices.index,
        recent_prices.values,
        color=RECENT_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=(
            f"Most Recent "
            f"{validated_config.recent_years} Years"
        ),
    )

    format_dashboard_date_axis(
        axes
    )

    apply_standard_formatting(
        axes.figure,
        axes,
        title=(
            DASHBOARD_PANEL_TITLES[
                PANEL_RECENT_PRICE
            ]
        ),
        subtitle=(
            f"Most recent "
            f"{validated_config.recent_years} years"
        ),
        x_label=DASHBOARD_DATE_LABEL,
        y_label=DASHBOARD_PRICE_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_currency_axis(
        axes
    )

    return axes


def populate_nominal_vs_real_panel(
    axes: Axes,
    data: pd.DataFrame,
    config: DashboardConfig,
) -> Axes:
    """
    Populate the nominal-versus-real price panel.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    nominal_prices = prepare_price_series(
        data=validated_data,
        column=(
            validated_config.price_column
        ),
    )

    real_prices = calculate_real_price(
        data=validated_data,
        price_column=(
            validated_config.price_column
        ),
    )

    comparison = pd.concat(
        [
            nominal_prices.rename(
                DASHBOARD_NOMINAL_LABEL
            ),
            real_prices.rename(
                DASHBOARD_REAL_LABEL
            ),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if comparison.empty:
        raise ValueError(
            "No nominal and inflation-adjusted "
            "prices are available for comparison."
        )

    axes.clear()

    axes.plot(
        comparison.index,
        comparison[
            DASHBOARD_NOMINAL_LABEL
        ],
        color=NOMINAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DASHBOARD_NOMINAL_LABEL,
    )

    axes.plot(
        comparison.index,
        comparison[
            DASHBOARD_REAL_LABEL
        ],
        color=REAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=DASHBOARD_REAL_LABEL,
    )

    format_dashboard_date_axis(
        axes
    )

    apply_standard_formatting(
        axes.figure,
        axes,
        title=(
            DASHBOARD_PANEL_TITLES[
                PANEL_NOMINAL_VS_REAL
            ]
        ),
        subtitle=(
            "Market price compared with "
            "inflation-adjusted purchasing power"
        ),
        x_label=DASHBOARD_DATE_LABEL,
        y_label=DASHBOARD_PRICE_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_currency_axis(
        axes
    )

    return axes


def populate_price_panels(
    axes: dict[str, Axes],
    data: pd.DataFrame,
    config: DashboardConfig,
) -> dict[str, Axes]:
    """
    Populate all price and inflation dashboard panels.
    """

    if not isinstance(
        axes,
        dict,
    ):
        raise TypeError(
            "axes must be a dictionary."
        )

    required_panels = {
        PANEL_HISTORICAL_PRICE,
        PANEL_RECENT_PRICE,
        PANEL_NOMINAL_VS_REAL,
    }

    missing_panels = (
        required_panels
        - set(axes)
    )

    if missing_panels:
        missing_text = ", ".join(
            sorted(missing_panels)
        )

        raise ValueError(
            "Dashboard axes are missing price panels: "
            f"{missing_text}."
        )

    populate_historical_panel(
        axes=axes[
            PANEL_HISTORICAL_PRICE
        ],
        data=data,
        config=config,
    )

    populate_recent_panel(
        axes=axes[
            PANEL_RECENT_PRICE
        ],
        data=data,
        config=config,
    )

    populate_nominal_vs_real_panel(
        axes=axes[
            PANEL_NOMINAL_VS_REAL
        ],
        data=data,
        config=config,
    )

    return axes

def populate_monthly_returns_panel(
    axes: Axes,
    data: pd.DataFrame,
    config: DashboardConfig,
) -> Axes:
    """
    Populate the monthly-returns dashboard panel.

    Positive and negative monthly returns are plotted as
    separate semantic series so changes in direction remain
    visually clear.

    Args:
        axes:
            Dashboard Axes receiving monthly returns.

        data:
            Validated dashboard data.

        config:
            Dashboard configuration.

    Returns:
        The populated Axes.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    monthly_returns = (
        calculate_monthly_returns(
            data=validated_data,
            column=(
                validated_config.price_column
            ),
        )
    )

    if monthly_returns.empty:
        raise ValueError(
            "No monthly returns are available "
            "for the dashboard panel."
        )

    positive_returns = (
        monthly_returns.where(
            monthly_returns >= 0
        )
    )

    negative_returns = (
        monthly_returns.where(
            monthly_returns < 0
        )
    )

    axes.clear()

    axes.plot(
        positive_returns.index,
        positive_returns.values,
        color=POSITIVE_RETURN_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=(
            DASHBOARD_POSITIVE_MONTHLY_RETURN_LABEL
        ),
    )

    axes.plot(
        negative_returns.index,
        negative_returns.values,
        color=NEGATIVE_RETURN_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=(
            DASHBOARD_NEGATIVE_MONTHLY_RETURN_LABEL
        ),
    )

    axes.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle="--",
        label=DASHBOARD_ZERO_RETURN_LABEL,
    )

    format_dashboard_date_axis(
        axes
    )

    apply_standard_formatting(
        axes.figure,
        axes,
        title=(
            DASHBOARD_PANEL_TITLES[
                PANEL_MONTHLY_RETURNS
            ]
        ),
        subtitle=(
            "Month-over-month percentage change "
            "in gold prices"
        ),
        x_label=DASHBOARD_DATE_LABEL,
        y_label=DASHBOARD_RETURN_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_percentage_axis(
        axes
    )

    return axes

def populate_annual_returns_panel(
    axes: Axes,
    data: pd.DataFrame,
    config: DashboardConfig,
) -> Axes:
    """
    Populate the annual-returns dashboard panel.

    Positive years use the shared positive-return color and
    negative years use the shared negative-return color.

    Args:
        axes:
            Dashboard Axes receiving annual returns.

        data:
            Validated dashboard data.

        config:
            Dashboard configuration.

    Returns:
        The populated Axes.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    annual_returns = (
        calculate_annual_returns(
            data=validated_data,
            column=(
                validated_config.price_column
            ),
        )
    )

    if annual_returns.empty:
        raise ValueError(
            "No annual returns are available "
            "for the dashboard panel."
        )

    bar_colors = [
        (
            POSITIVE_RETURN_COLOR
            if return_value >= 0
            else NEGATIVE_RETURN_COLOR
        )
        for return_value in (
            annual_returns.values
        )
    ]

    axes.clear()

    axes.bar(
        annual_returns.index,
        annual_returns.values,
        color=bar_colors,
        label=DASHBOARD_ANNUAL_RETURN_LABEL,
    )

    axes.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle="--",
    )

    apply_standard_formatting(
        axes.figure,
        axes,
        title=(
            DASHBOARD_PANEL_TITLES[
                PANEL_ANNUAL_RETURNS
            ]
        ),
        subtitle=(
            "Calendar-year returns based on "
            "year-end gold prices"
        ),
        x_label=DASHBOARD_YEAR_LABEL,
        y_label=DASHBOARD_RETURN_LABEL,
        show_grid=True,
        show_legend=False,
        apply_tight_layout=False,
    )

    format_percentage_axis(
        axes
    )

    return axes

def populate_returns_panels(
    axes: dict[str, Axes],
    data: pd.DataFrame,
    config: DashboardConfig,
) -> dict[str, Axes]:
    """
    Populate all return-analysis dashboard panels.

    Args:
        axes:
            Dashboard panel mapping.

        data:
            Validated dashboard data.

        config:
            Dashboard configuration.

    Returns:
        The populated dashboard panel mapping.
    """

    if not isinstance(
        axes,
        dict,
    ):
        raise TypeError(
            "axes must be a dictionary."
        )

    required_panels = {
        PANEL_MONTHLY_RETURNS,
        PANEL_ANNUAL_RETURNS,
    }

    missing_panels = (
        required_panels
        - set(axes)
    )

    if missing_panels:
        missing_text = ", ".join(
            sorted(missing_panels)
        )

        raise ValueError(
            "Dashboard axes are missing return panels: "
            f"{missing_text}."
        )

    populate_monthly_returns_panel(
        axes=axes[
            PANEL_MONTHLY_RETURNS
        ],
        data=data,
        config=config,
    )

    populate_annual_returns_panel(
        axes=axes[
            PANEL_ANNUAL_RETURNS
        ],
        data=data,
        config=config,
    )

    return axes

# ------------------------------------------------------------------
# Result validation
# ------------------------------------------------------------------


def validate_dashboard_result(
    result: DashboardResult,
) -> DashboardResult:
    """
    Validate a completed DashboardResult.
    """

    if not isinstance(
        result,
        DashboardResult,
    ):
        raise TypeError(
            "result must be a DashboardResult."
        )

    if not isinstance(
        result.figure,
        Figure,
    ):
        raise TypeError(
            "result.figure must be a matplotlib Figure."
        )

    if not isinstance(
        result.axes,
        dict,
    ):
        raise TypeError(
            "result.axes must be a dictionary."
        )

    missing_panels = (
        set(DASHBOARD_PANEL_ORDER)
        - set(result.axes)
    )

    if missing_panels:
        missing_text = ", ".join(
            sorted(missing_panels)
        )

        raise ValueError(
            "Dashboard result is missing panels: "
            f"{missing_text}."
        )

    for panel_name, panel_axes in (
        result.axes.items()
    ):
        if not isinstance(
            panel_name,
            str,
        ):
            raise TypeError(
                "Dashboard panel names must be strings."
            )

        if not isinstance(
            panel_axes,
            Axes,
        ):
            raise TypeError(
                f"Dashboard panel {panel_name!r} "
                "must contain a matplotlib Axes."
            )

    validate_dashboard_config(
        result.config
    )

    if not isinstance(
        result.metadata,
        dict,
    ):
        raise TypeError(
            "result.metadata must be a dictionary."
        )

    return result

def populate_forecast_panel(
    axes: Axes,
    data: pd.DataFrame,
    config: DashboardConfig,
) -> Axes:
    """
    Populate the hypothetical forecast dashboard panel.

    The panel displays recent historical prices followed by a
    deterministic compound-growth projection. The forecast is
    illustrative and is not a market prediction.

    Args:
        axes:
            Dashboard Axes receiving the forecast chart.

        data:
            Validated dashboard data.

        config:
            Dashboard configuration.

    Returns:
        The populated Axes.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    prices = prepare_price_series(
        data=validated_data,
        column=(
            validated_config.price_column
        ),
    )

    valid_prices = (
        prices
        .dropna()
        .sort_index()
    )

    if valid_prices.empty:
        raise ValueError(
            "No historical prices are available "
            "for the forecast dashboard panel."
        )

    if (
        validated_config.forecast_history_years
        is None
    ):
        visible_history = valid_prices
    else:
        history_start_date = (
            valid_prices.index[-1]
            - pd.DateOffset(
                years=(
                    validated_config
                    .forecast_history_years
                )
            )
        )

        visible_history = valid_prices.loc[
            valid_prices.index
            >= history_start_date
        ]

    if visible_history.empty:
        raise ValueError(
            "No historical prices are available "
            "inside the configured forecast-history period."
        )

    forecast = generate_forecast_series(
        price_series=valid_prices,
        annual_growth_rate=(
            validated_config.annual_growth_rate
        ),
        years=(
            validated_config.forecast_years
        ),
    )

    if forecast.empty:
        raise ValueError(
            "No forecast values are available "
            "for the dashboard panel."
        )

    forecast_boundary = (
        valid_prices.index[-1]
    )

    connected_forecast = pd.concat(
        [
            valid_prices.iloc[-1:].rename(
                forecast.name
            ),
            forecast,
        ]
    )

    axes.clear()

    axes.plot(
        visible_history.index,
        visible_history.values,
        color=HISTORICAL_PRICE_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=(
            DASHBOARD_FORECAST_HISTORY_LABEL
        ),
    )

    axes.plot(
        connected_forecast.index,
        connected_forecast.values,
        color=FORECAST_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        linestyle=(
            DASHBOARD_FORECAST_LINE_STYLE
        ),
        marker="o",
        markersize=(
            DASHBOARD_FORECAST_MARKER_SIZE
        ),
        markevery=[
            len(connected_forecast) - 1
        ],
        label=(
            f"{DASHBOARD_FORECAST_LABEL} "
            f"({validated_config.annual_growth_rate:.1%} annually)"
        ),
    )

    annotate_vertical_event(
        axes,
        x_value=forecast_boundary,
        label=(
            DASHBOARD_FORECAST_BOUNDARY_LABEL
        ),
        line_color=FORECAST_BOUNDARY_COLOR,
        line_style=(
            DASHBOARD_FORECAST_BOUNDARY_STYLE
        ),
    )

    format_dashboard_date_axis(
        axes
    )

    apply_standard_formatting(
        axes.figure,
        axes,
        title=(
            DASHBOARD_PANEL_TITLES[
                PANEL_FORECAST
            ]
        ),
        subtitle=(
            f"{validated_config.forecast_years}-year "
            "hypothetical projection using a constant "
            f"{validated_config.annual_growth_rate:.1%} "
            "annual growth assumption"
        ),
        x_label=DASHBOARD_DATE_LABEL,
        y_label=DASHBOARD_PRICE_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_currency_axis(
        axes
    )

    if validated_config.show_annotations:
        annotate_latest_value(
            axes,
            forecast,
            prefix=(
                f"Projected "
                f"{validated_config.forecast_years}-Year Value"
            ),
            value_format="currency",
            marker_color=FORECAST_COLOR,
            offset=(
                -150,
                16,
            ),
        )

    if (
        validated_config
        .show_forecast_disclaimer
    ):
        axes.text(
            DASHBOARD_FORECAST_DISCLAIMER_X,
            DASHBOARD_FORECAST_DISCLAIMER_Y,
            DASHBOARD_FORECAST_DISCLAIMER,
            transform=axes.transAxes,
            fontsize=SUBTITLE_FONT_SIZE,
            fontfamily=DEFAULT_FONT_FAMILY,
            color=SECONDARY_TEXT_COLOR,
            horizontalalignment="left",
            verticalalignment="bottom",
        )

    return axes

def populate_forecast_panels(
    axes: dict[str, Axes],
    data: pd.DataFrame,
    config: DashboardConfig,
) -> dict[str, Axes]:
    """
    Populate the dashboard forecast panel.

    Args:
        axes:
            Dashboard panel mapping.

        data:
            Validated dashboard data.

        config:
            Dashboard configuration.

    Returns:
        The populated dashboard panel mapping.
    """

    if not isinstance(
        axes,
        dict,
    ):
        raise TypeError(
            "axes must be a dictionary."
        )

    if PANEL_FORECAST not in axes:
        raise ValueError(
            "Dashboard axes are missing "
            "the forecast panel."
        )

    populate_forecast_panel(
        axes=axes[
            PANEL_FORECAST
        ],
        data=data,
        config=config,
    )

    return axes

# ------------------------------------------------------------------
# Main dashboard builder
# ------------------------------------------------------------------


def build_dashboard_layout(
    data: pd.DataFrame,
    config: DashboardConfig | None = None,
) -> DashboardResult:
    """
    Build and populate the Gold Value Predictor dashboard.

    The completed dashboard includes:

    - Summary metrics
    - Historical gold prices
    - Recent gold prices
    - Nominal versus inflation-adjusted prices
    - Monthly returns
    - Annual returns
    - Rolling volatility
    - Rolling drawdown
    - Hypothetical forecast

    Args:
        data:
            Source dashboard DataFrame.

        config:
            Optional dashboard configuration.

    Returns:
        A validated DashboardResult containing the completed
        dashboard Figure and named panel Axes.
    """

    context = create_dashboard_context(
        data=data,
        config=config,
    )

    figure = create_dashboard_figure(
        context.config
    )

    grid = create_dashboard_grid(
        figure
    )

    axes = create_dashboard_axes(
        figure,
        grid,
    )

    add_dashboard_heading(
        figure,
        context.config,
    )

    style_dashboard_axes(
        axes
    )

    summary = build_dashboard_summary(
        data=context.data,
        config=context.config,
    )

    populate_summary_panel(
        axes=axes[
            PANEL_SUMMARY
        ],
        summary=summary,
        config=context.config,
    )

    populate_price_panels(
        axes=axes,
        data=context.data,
        config=context.config,
    )

    populate_returns_panels(
        axes=axes,
        data=context.data,
        config=context.config,
    )

    populate_rolling_panels(
        axes=axes,
        data=context.data,
        config=context.config,
    )

    populate_forecast_panels(
        axes=axes,
        data=context.data,
        config=context.config,
    )

    result = DashboardResult(
        figure=figure,
        axes=axes,
        config=context.config,
        metadata=context.metadata,
    )

    return validate_dashboard_result(
        result
    )

def populate_rolling_volatility_panel(
    axes: Axes,
    data: pd.DataFrame,
    config: DashboardConfig,
) -> Axes:
    """
    Populate the rolling-volatility dashboard panel.

    Volatility is calculated from monthly returns using the
    configured rolling window and displayed on an annualized
    basis.

    Args:
        axes:
            Dashboard Axes receiving rolling volatility.

        data:
            Validated dashboard data.

        config:
            Dashboard configuration.

    Returns:
        The populated Axes.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    rolling_volatility = (
        calculate_rolling_volatility(
            data=validated_data,
            column=(
                validated_config.price_column
            ),
            window=(
                validated_config.rolling_window
            ),
            annualize=True,
        )
    )

    valid_volatility = (
        rolling_volatility.dropna()
    )

    if valid_volatility.empty:
        raise ValueError(
            "No rolling volatility values are "
            "available for the dashboard panel."
        )

    axes.clear()

    axes.plot(
        valid_volatility.index,
        valid_volatility.values,
        color=VOLATILITY_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=(
            f"{validated_config.rolling_window}-Month "
            f"{DASHBOARD_VOLATILITY_LABEL}"
        ),
    )

    format_dashboard_date_axis(
        axes
    )

    apply_standard_formatting(
        axes.figure,
        axes,
        title=(
            DASHBOARD_PANEL_TITLES[
                PANEL_ROLLING_VOLATILITY
            ]
        ),
        subtitle=(
            f"Annualized volatility across a "
            f"{validated_config.rolling_window}-month "
            "rolling window"
        ),
        x_label=DASHBOARD_DATE_LABEL,
        y_label=DASHBOARD_PERCENTAGE_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="upper left",
        apply_tight_layout=False,
    )

    format_percentage_axis(
        axes
    )

    return axes

def populate_rolling_drawdown_panel(
    axes: Axes,
    data: pd.DataFrame,
    config: DashboardConfig,
) -> Axes:
    """
    Populate the rolling-drawdown dashboard panel.

    Drawdown measures the decline from the highest observed
    price inside each configured rolling window.

    Args:
        axes:
            Dashboard Axes receiving rolling drawdown.

        data:
            Validated dashboard data.

        config:
            Dashboard configuration.

    Returns:
        The populated Axes.
    """

    if not isinstance(
        axes,
        Axes,
    ):
        raise TypeError(
            "axes must be a matplotlib Axes."
        )

    validated_config = (
        validate_dashboard_config(
            config
        )
    )

    validated_data = (
        validate_dashboard_data(
            data,
            price_column=(
                validated_config.price_column
            ),
        )
    )

    rolling_drawdown = (
        calculate_rolling_drawdown(
            data=validated_data,
            column=(
                validated_config.price_column
            ),
            window=(
                validated_config.rolling_window
            ),
        )
    )

    valid_drawdown = (
        rolling_drawdown.dropna()
    )

    if valid_drawdown.empty:
        raise ValueError(
            "No rolling drawdown values are "
            "available for the dashboard panel."
        )

    axes.clear()

    axes.plot(
        valid_drawdown.index,
        valid_drawdown.values,
        color=DRAWDOWN_COLOR,
        linewidth=DEFAULT_LINE_WIDTH,
        label=(
            f"{validated_config.rolling_window}-Month "
            f"{DASHBOARD_DRAWDOWN_LABEL}"
        ),
    )

    axes.fill_between(
        valid_drawdown.index,
        valid_drawdown.values,
        0,
        color=DRAWDOWN_COLOR,
        alpha=DASHBOARD_DRAWDOWN_FILL_ALPHA,
    )

    axes.axhline(
        y=0,
        color=REFERENCE_LINE_COLOR,
        linewidth=REFERENCE_LINE_WIDTH,
        linestyle="--",
        label=DASHBOARD_ROLLING_HIGH_LABEL,
    )

    format_dashboard_date_axis(
        axes
    )

    apply_standard_formatting(
        axes.figure,
        axes,
        title=(
            DASHBOARD_PANEL_TITLES[
                PANEL_ROLLING_DRAWDOWN
            ]
        ),
        subtitle=(
            f"Decline from the rolling high across a "
            f"{validated_config.rolling_window}-month window"
        ),
        x_label=DASHBOARD_DATE_LABEL,
        y_label=DASHBOARD_PERCENTAGE_LABEL,
        show_grid=True,
        show_legend=True,
        legend_location="lower left",
        apply_tight_layout=False,
    )

    format_percentage_axis(
        axes
    )

    return axes

def populate_rolling_panels(
    axes: dict[str, Axes],
    data: pd.DataFrame,
    config: DashboardConfig,
) -> dict[str, Axes]:
    """
    Populate all rolling-analysis dashboard panels.

    Args:
        axes:
            Dashboard panel mapping.

        data:
            Validated dashboard data.

        config:
            Dashboard configuration.

    Returns:
        The populated dashboard panel mapping.
    """

    if not isinstance(
        axes,
        dict,
    ):
        raise TypeError(
            "axes must be a dictionary."
        )

    required_panels = {
        PANEL_ROLLING_VOLATILITY,
        PANEL_ROLLING_DRAWDOWN,
    }

    missing_panels = (
        required_panels
        - set(axes)
    )

    if missing_panels:
        missing_text = ", ".join(
            sorted(missing_panels)
        )

        raise ValueError(
            "Dashboard axes are missing rolling panels: "
            f"{missing_text}."
        )

    populate_rolling_volatility_panel(
        axes=axes[
            PANEL_ROLLING_VOLATILITY
        ],
        data=data,
        config=config,
    )

    populate_rolling_drawdown_panel(
        axes=axes[
            PANEL_ROLLING_DRAWDOWN
        ],
        data=data,
        config=config,
    )

    return axes