"""
Tests for the Gold Value Predictor dashboard architecture.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.ticker import FuncFormatter
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from visualizations.dashboard import (
    DASHBOARD_PANEL_ORDER,
    DEFAULT_DASHBOARD_FIGURE_SIZE,
    DEFAULT_DASHBOARD_TITLE,
    DashboardConfig,
    DashboardResult,
    build_dashboard_metadata,
    create_dashboard_context,
    validate_dashboard_config,
    validate_dashboard_data,
    validate_dashboard_figure_size,
    validate_dashboard_result,
    validate_dashboard_text,
    validate_positive_integer,
    DASHBOARD_PANEL_TITLES,
    PANEL_FORECAST,
    PANEL_HISTORICAL_PRICE,
    PANEL_SUMMARY,
    add_dashboard_heading,
    build_dashboard_layout,
    create_dashboard_axes,
    create_dashboard_figure,
    create_dashboard_grid,
    style_dashboard_axes,
    style_dashboard_panel,
    DASHBOARD_SUMMARY_LABELS,
    DASHBOARD_SUMMARY_ORDER,
    DashboardSummary,
    build_dashboard_summary,
    format_dashboard_currency,
    format_dashboard_date,
    format_dashboard_percentage,
    format_dashboard_summary_values,
    populate_summary_panel,
    SUMMARY_LATEST_PRICE,
    format_dashboard_date_axis,
    populate_historical_panel,
    populate_nominal_vs_real_panel,
    populate_price_panels,
    populate_recent_panel,
    DASHBOARD_HISTORICAL_LABEL,
    DASHBOARD_NOMINAL_LABEL,
    DASHBOARD_REAL_LABEL,
    PANEL_RECENT_PRICE,
    PANEL_NOMINAL_VS_REAL,
    DASHBOARD_ANNUAL_RETURN_LABEL,
    DASHBOARD_NEGATIVE_MONTHLY_RETURN_LABEL,
    DASHBOARD_POSITIVE_MONTHLY_RETURN_LABEL,
    DASHBOARD_ZERO_RETURN_LABEL,
    PANEL_ANNUAL_RETURNS,
    PANEL_MONTHLY_RETURNS,
    populate_annual_returns_panel,
    populate_monthly_returns_panel,
    populate_returns_panels,
    DASHBOARD_DRAWDOWN_LABEL,
    DASHBOARD_ROLLING_HIGH_LABEL,
    DASHBOARD_VOLATILITY_LABEL,
    PANEL_ROLLING_DRAWDOWN,
    PANEL_ROLLING_VOLATILITY,
    populate_rolling_drawdown_panel,
    populate_rolling_panels,
    populate_rolling_volatility_panel,
    DASHBOARD_FORECAST_BOUNDARY_LABEL,
    DASHBOARD_FORECAST_DISCLAIMER,
    DASHBOARD_FORECAST_HISTORY_LABEL,
    DASHBOARD_FORECAST_LABEL,
    PANEL_FORECAST,
    populate_forecast_panel,
    populate_forecast_panels,
)

from visualizations.styling import (
    NEGATIVE_RETURN_COLOR,
    POSITIVE_RETURN_COLOR,
    DRAWDOWN_COLOR,
    VOLATILITY_COLOR,
    FORECAST_BOUNDARY_COLOR,
    FORECAST_COLOR,
    HISTORICAL_PRICE_COLOR,
)

import visualizations.dashboard as dashboard_module

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture(autouse=True)
def close_figures_after_test():
    """
    Close all Matplotlib figures after every dashboard test.
    """

    yield

    plt.close("all")


@pytest.fixture
def dashboard_data() -> pd.DataFrame:
    """
    Create valid monthly dashboard data.
    """

    return pd.DataFrame(
        {
            "Date": pd.date_range(
                start="2020-01-01",
                periods=36,
                freq="MS",
            ),
            "Gold Price": [
                1400.0 + month * 10.0
                for month in range(36)
            ],
            "CPI": [
                250.0 + month * 0.4
                for month in range(36)
            ],
        }
    )


# ------------------------------------------------------------------
# Dashboard test fixtures
# ------------------------------------------------------------------


@pytest.fixture
def dashboard_data() -> pd.DataFrame:
    """
    Create valid monthly dashboard data.

    The prices rise steadily so the fixture is predictable and
    useful for dashboard structure, validation, summary, price,
    and inflation-panel tests.
    """

    dates = pd.date_range(
        start="2020-01-01",
        periods=36,
        freq="MS",
    )

    prices = [
        1400.0 + month * 10.0
        for month in range(36)
    ]

    cpi_values = [
        250.0 + month * 0.4
        for month in range(36)
    ]

    return pd.DataFrame(
        {
            "Date": dates,
            "Gold Price": prices,
            "CPI": cpi_values,
        }
    )


@pytest.fixture
def mixed_dashboard_data() -> pd.DataFrame:
    """
    Create monthly dashboard data containing both gains and losses.

    This fixture supports return-chart tests that require positive
    and negative monthly or annual return values.
    """

    dates = pd.date_range(
        start="2018-01-01",
        periods=60,
        freq="MS",
    )

    prices = (
        list(
            range(
                100,
                112,
            )
        )
        + list(
            range(
                112,
                100,
                -1,
            )
        )
        + list(
            range(
                101,
                113,
            )
        )
        + list(
            range(
                113,
                101,
                -1,
            )
        )
        + list(
            range(
                102,
                114,
            )
        )
    )

    cpi_values = [
        250.0 + month * 0.3
        for month in range(60)
    ]

    return pd.DataFrame(
        {
            "Date": dates,
            "Gold Price": prices,
            "CPI": cpi_values,
        }
    )


# ------------------------------------------------------------------
# Default configuration tests
# ------------------------------------------------------------------


def test_dashboard_config_uses_expected_defaults() -> None:
    config = DashboardConfig()

    assert (
        config.title
        == DEFAULT_DASHBOARD_TITLE
    )

    assert (
        config.figure_size
        == DEFAULT_DASHBOARD_FIGURE_SIZE
    )

    assert config.recent_years == 5
    assert config.rolling_window == 12
    assert config.forecast_years == 5
    assert config.annual_growth_rate == 0.05
    assert config.show_annotations is True


def test_dashboard_config_is_immutable() -> None:
    config = DashboardConfig()

    with pytest.raises(
        AttributeError,
    ):
        config.title = "Changed"  # type: ignore[misc]


def test_dashboard_panel_order_contains_unique_names() -> None:
    assert len(
        DASHBOARD_PANEL_ORDER
    ) == len(
        set(DASHBOARD_PANEL_ORDER)
    )


def test_dashboard_panel_order_includes_summary_and_forecast() -> None:
    assert "summary" in DASHBOARD_PANEL_ORDER
    assert "forecast" in DASHBOARD_PANEL_ORDER


# ------------------------------------------------------------------
# Text validation tests
# ------------------------------------------------------------------


def test_validate_dashboard_text_accepts_text() -> None:
    result = validate_dashboard_text(
        "Dashboard",
        "title",
        allow_none=False,
    )

    assert result == "Dashboard"


def test_validate_dashboard_text_accepts_none_when_allowed() -> None:
    assert (
        validate_dashboard_text(
            None,
            "subtitle",
        )
        is None
    )


def test_validate_dashboard_text_rejects_none_when_required() -> None:
    with pytest.raises(
        TypeError,
        match="title must be a string",
    ):
        validate_dashboard_text(
            None,
            "title",
            allow_none=False,
        )


def test_validate_dashboard_text_rejects_blank_text() -> None:
    with pytest.raises(
        ValueError,
        match="title cannot be empty",
    ):
        validate_dashboard_text(
            "   ",
            "title",
            allow_none=False,
        )


# ------------------------------------------------------------------
# Integer and figure-size validation tests
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        1,
        12,
        300,
    ],
)
def test_validate_positive_integer_accepts_valid_values(
    value: int,
) -> None:
    assert (
        validate_positive_integer(
            value,
            "value",
        )
        == value
    )


@pytest.mark.parametrize(
    "value",
    [
        True,
        1.5,
        "12",
        None,
    ],
)
def test_validate_positive_integer_rejects_nonintegers(
    value: object,
) -> None:
    with pytest.raises(
        TypeError,
        match="value must be an integer",
    ):
        validate_positive_integer(
            value,  # type: ignore[arg-type]
            "value",
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
    ],
)
def test_validate_positive_integer_rejects_nonpositive_values(
    value: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="value must be at least 1",
    ):
        validate_positive_integer(
            value,
            "value",
        )


def test_validate_dashboard_figure_size_normalizes_values() -> None:
    result = validate_dashboard_figure_size(
        (
            18,
            24,
        )
    )

    assert result == (
        18.0,
        24.0,
    )


@pytest.mark.parametrize(
    "figure_size",
    [
        [18, 24],
        (18,),
        (18, 24, 30),
        ("18", 24),
        (True, 24),
    ],
)
def test_validate_dashboard_figure_size_rejects_invalid_values(
    figure_size: object,
) -> None:
    with pytest.raises(TypeError):
        validate_dashboard_figure_size(
            figure_size,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "figure_size",
    [
        (0, 24),
        (18, 0),
        (-18, 24),
    ],
)
def test_validate_dashboard_figure_size_rejects_nonpositive_values(
    figure_size: tuple[float, float],
) -> None:
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        validate_dashboard_figure_size(
            figure_size
        )


# ------------------------------------------------------------------
# Configuration validation tests
# ------------------------------------------------------------------


def test_validate_dashboard_config_returns_config() -> None:
    config = DashboardConfig()

    result = validate_dashboard_config(
        config
    )

    assert result is config


def test_validate_dashboard_config_rejects_invalid_type() -> None:
    with pytest.raises(
        TypeError,
        match="config must be a DashboardConfig",
    ):
        validate_dashboard_config(
            "invalid"  # type: ignore[arg-type]
        )


def test_validate_dashboard_config_accepts_custom_values() -> None:
    config = DashboardConfig(
        title="Custom Dashboard",
        subtitle=None,
        recent_years=3,
        rolling_window=6,
        forecast_years=2,
        forecast_history_years=None,
        annual_growth_rate=0.04,
        figure_size=(
            16.0,
            20.0,
        ),
        dpi=150,
        source_note=None,
        watermark=None,
        show_annotations=False,
        show_forecast_disclaimer=False,
    )

    result = validate_dashboard_config(
        config
    )

    assert result is config


def test_validate_dashboard_config_rejects_invalid_growth_rate() -> None:
    config = DashboardConfig(
        annual_growth_rate=-1.0,
    )

    with pytest.raises(
        ValueError,
        match=(
            "annual_growth_rate must be "
            "greater than -1"
        ),
    ):
        validate_dashboard_config(
            config
        )


def test_validate_dashboard_config_rejects_nonboolean_option() -> None:
    config = DashboardConfig(
        show_annotations="yes",  # type: ignore[arg-type]
    )

    with pytest.raises(
        TypeError,
        match=(
            "show_annotations must be "
            "a Boolean value"
        ),
    ):
        validate_dashboard_config(
            config
        )


# ------------------------------------------------------------------
# Data validation tests
# ------------------------------------------------------------------


def test_validate_dashboard_data_returns_sorted_copy(
    dashboard_data: pd.DataFrame,
) -> None:
    reversed_data = dashboard_data.iloc[
        ::-1
    ].reset_index(drop=True)

    result = validate_dashboard_data(
        reversed_data
    )

    assert result is not reversed_data

    assert result[
        "Date"
    ].is_monotonic_increasing


def test_validate_dashboard_data_rejects_nondataframe() -> None:
    with pytest.raises(
        TypeError,
        match=(
            "data must be a Pandas DataFrame"
        ),
    ):
        validate_dashboard_data(
            []  # type: ignore[arg-type]
        )


def test_validate_dashboard_data_rejects_empty_data() -> None:
    with pytest.raises(
        ValueError,
        match="Dashboard data cannot be empty",
    ):
        validate_dashboard_data(
            pd.DataFrame()
        )


def test_validate_dashboard_data_rejects_missing_date(
    dashboard_data: pd.DataFrame,
) -> None:
    data = dashboard_data.drop(
        columns=["Date"]
    )

    with pytest.raises(
        ValueError,
        match="Date",
    ):
        validate_dashboard_data(data)


def test_validate_dashboard_data_rejects_missing_price(
    dashboard_data: pd.DataFrame,
) -> None:
    data = dashboard_data.drop(
        columns=["Gold Price"]
    )

    with pytest.raises(
        ValueError,
        match="Gold Price",
    ):
        validate_dashboard_data(data)


def test_validate_dashboard_data_rejects_invalid_dates(
    dashboard_data: pd.DataFrame,
) -> None:
    data = dashboard_data.copy()

    data["Date"] = data[
        "Date"
    ].astype("object")

    data.loc[
        0,
        "Date",
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match="invalid Date values",
    ):
        validate_dashboard_data(
            data
        )


def test_validate_dashboard_data_rejects_unusable_prices(
    dashboard_data: pd.DataFrame,
) -> None:
    data = dashboard_data.copy()

    data[
        "Gold Price"
    ] = "invalid"

    with pytest.raises(
        ValueError,
        match="no usable values",
    ):
        validate_dashboard_data(data)


# ------------------------------------------------------------------
# Metadata and context tests
# ------------------------------------------------------------------


def test_build_dashboard_metadata_returns_expected_values(
    dashboard_data: pd.DataFrame,
) -> None:
    config = DashboardConfig()

    metadata = build_dashboard_metadata(
        dashboard_data,
        config,
    )

    assert metadata[
        "row_count"
    ] == 36

    assert metadata[
        "column_count"
    ] == 3

    assert metadata[
        "latest_price"
    ] == pytest.approx(
        1750.0
    )

    assert metadata[
        "panel_count"
    ] == len(
        DASHBOARD_PANEL_ORDER
    )


def test_create_dashboard_context_uses_default_config(
    dashboard_data: pd.DataFrame,
) -> None:
    context = create_dashboard_context(
        dashboard_data
    )

    assert isinstance(
        context.config,
        DashboardConfig,
    )

    assert context.metadata[
        "row_count"
    ] == 36

    assert context.data[
        "Date"
    ].is_monotonic_increasing


def test_create_dashboard_context_uses_custom_config(
    dashboard_data: pd.DataFrame,
) -> None:
    config = DashboardConfig(
        recent_years=3,
        rolling_window=6,
    )

    context = create_dashboard_context(
        dashboard_data,
        config=config,
    )

    assert context.config is config

    assert context.metadata[
        "recent_years"
    ] == 3

    assert context.metadata[
        "rolling_window"
    ] == 6


# ------------------------------------------------------------------
# Result validation tests
# ------------------------------------------------------------------


def test_validate_dashboard_result_accepts_complete_result() -> None:
    figure = plt.figure()

    axes = {
        panel_name: figure.add_subplot(
            3,
            3,
            panel_index + 1,
        )
        for panel_index, panel_name in enumerate(
            DASHBOARD_PANEL_ORDER
        )
    }

    result = DashboardResult(
        figure=figure,
        axes=axes,
        config=DashboardConfig(),
        metadata={
            "example": True,
        },
    )

    assert (
        validate_dashboard_result(
            result
        )
        is result
    )


def test_validate_dashboard_result_rejects_missing_panels() -> None:
    figure, axes = plt.subplots()

    result = DashboardResult(
        figure=figure,
        axes={
            "summary": axes,
        },
        config=DashboardConfig(),
        metadata={},
    )

    with pytest.raises(
        ValueError,
        match="missing panels",
    ):
        validate_dashboard_result(
            result
        )


def test_dashboard_result_axes_are_axes_objects() -> None:
    figure = plt.figure()

    axes = {
        panel_name: figure.add_subplot(
            3,
            3,
            panel_index + 1,
        )
        for panel_index, panel_name in enumerate(
            DASHBOARD_PANEL_ORDER
        )
    }

    result = DashboardResult(
        figure=figure,
        axes=axes,
        config=DashboardConfig(),
        metadata={},
    )

    assert all(
        isinstance(
            panel_axes,
            Axes,
        )
        for panel_axes in result.axes.values()
    )

# ------------------------------------------------------------------
# Dashboard layout tests
# ------------------------------------------------------------------


def test_create_dashboard_figure_uses_configured_size() -> None:
    config = DashboardConfig(
        figure_size=(
            16.0,
            20.0,
        ),
        dpi=150,
    )

    figure = create_dashboard_figure(
        config
    )

    width, height = (
        figure.get_size_inches()
    )

    assert width == pytest.approx(
        16.0
    )

    assert height == pytest.approx(
        20.0
    )

    assert figure.dpi == pytest.approx(
        150
    )


def test_create_dashboard_grid_returns_gridspec() -> None:
    figure = plt.figure()

    grid = create_dashboard_grid(
        figure
    )

    assert isinstance(
        grid,
        GridSpec,
    )

    assert grid.nrows == 6
    assert grid.ncols == 2


def test_create_dashboard_axes_returns_all_panels() -> None:
    figure = plt.figure()

    grid = create_dashboard_grid(
        figure
    )

    axes = create_dashboard_axes(
        figure,
        grid,
    )

    assert set(
        axes
    ) == set(
        DASHBOARD_PANEL_ORDER
    )

    assert all(
        isinstance(
            panel_axes,
            Axes,
        )
        for panel_axes in axes.values()
    )


def test_create_dashboard_axes_creates_expected_count() -> None:
    figure = plt.figure()

    grid = create_dashboard_grid(
        figure
    )

    axes = create_dashboard_axes(
        figure,
        grid,
    )

    assert len(
        axes
    ) == len(
        DASHBOARD_PANEL_ORDER
    )

    assert len(
        figure.axes
    ) == len(
        DASHBOARD_PANEL_ORDER
    )


def test_add_dashboard_heading_adds_title_and_subtitle() -> None:
    figure = plt.figure()

    config = DashboardConfig(
        title="Custom Dashboard",
        subtitle="Custom Subtitle",
    )

    add_dashboard_heading(
        figure,
        config,
    )

    text_values = [
        text.get_text()
        for text in figure.texts
    ]

    assert "Custom Dashboard" in text_values
    assert "Custom Subtitle" in text_values


def test_add_dashboard_heading_allows_no_subtitle() -> None:
    figure = plt.figure()

    config = DashboardConfig(
        subtitle=None,
    )

    add_dashboard_heading(
        figure,
        config,
    )

    text_values = [
        text.get_text()
        for text in figure.texts
    ]

    assert config.title in text_values

    assert len(
        figure.texts
    ) == 1


def test_style_dashboard_panel_sets_expected_title() -> None:
    _, axes = plt.subplots()

    result = style_dashboard_panel(
        axes,
        PANEL_HISTORICAL_PRICE,
    )

    assert result is axes

    assert (
        axes.get_title(
            loc="left"
        )
        == DASHBOARD_PANEL_TITLES[
            PANEL_HISTORICAL_PRICE
        ]
    )


def test_style_dashboard_summary_hides_ticks() -> None:
    _, axes = plt.subplots()

    style_dashboard_panel(
        axes,
        PANEL_SUMMARY,
    )

    assert not axes.get_xticks().size
    assert not axes.get_yticks().size


def test_style_dashboard_panel_rejects_unknown_panel() -> None:
    _, axes = plt.subplots()

    with pytest.raises(
        ValueError,
        match="Unknown dashboard panel",
    ):
        style_dashboard_panel(
            axes,
            "unknown",
        )


def test_style_dashboard_axes_styles_all_panels() -> None:
    figure = plt.figure()

    grid = create_dashboard_grid(
        figure
    )

    axes = create_dashboard_axes(
        figure,
        grid,
    )

    result = style_dashboard_axes(
        axes
    )

    assert result is axes

    for panel_name in (
        DASHBOARD_PANEL_ORDER
    ):
        assert (
            axes[
                panel_name
            ].get_title(
                loc="left"
            )
            == DASHBOARD_PANEL_TITLES[
                panel_name
            ]
        )


def test_build_dashboard_layout_returns_valid_result(
    dashboard_data: pd.DataFrame,
) -> None:
    result = build_dashboard_layout(
        dashboard_data
    )

    assert isinstance(
        result,
        DashboardResult,
    )

    assert isinstance(
        result.figure,
        Figure,
    )

    assert set(
        result.axes
    ) == set(
        DASHBOARD_PANEL_ORDER
    )


def test_build_dashboard_layout_uses_custom_config(
    dashboard_data: pd.DataFrame,
) -> None:
    config = DashboardConfig(
        title="Custom Dashboard",
        subtitle=None,
        figure_size=(
            16.0,
            20.0,
        ),
    )

    result = build_dashboard_layout(
        dashboard_data,
        config=config,
    )

    assert result.config is config

    figure_texts = [
        text.get_text()
        for text in result.figure.texts
    ]

    assert "Custom Dashboard" in figure_texts


def test_dashboard_full_width_panels_exist(
    dashboard_data: pd.DataFrame,
) -> None:
    result = build_dashboard_layout(
        dashboard_data
    )

    assert (
        PANEL_HISTORICAL_PRICE
        in result.axes
    )

    assert (
        PANEL_FORECAST
        in result.axes
    )

# ------------------------------------------------------------------
# Dashboard summary tests
# ------------------------------------------------------------------


def test_format_dashboard_currency() -> None:
    assert (
        format_dashboard_currency(
            1234.5
        )
        == "$1,234.50"
    )


def test_format_dashboard_percentage() -> None:
    assert (
        format_dashboard_percentage(
            0.0525
        )
        == "5.25%"
    )


def test_format_dashboard_date() -> None:
    date = pd.Timestamp(
        "2026-07-01"
    )

    assert (
        format_dashboard_date(
            date
        )
        == "Jul 2026"
    )


@pytest.mark.parametrize(
    "formatter",
    [
        format_dashboard_currency,
        format_dashboard_percentage,
    ],
)
def test_dashboard_numeric_formatters_reject_invalid_values(
    formatter,
) -> None:
    with pytest.raises(
        TypeError,
        match="value must be numeric",
    ):
        formatter(
            "invalid"
        )


def test_build_dashboard_summary_returns_summary(
    dashboard_data: pd.DataFrame,
) -> None:
    config = DashboardConfig(
        forecast_years=2,
        annual_growth_rate=0.05,
    )

    summary = build_dashboard_summary(
        dashboard_data,
        config,
    )

    assert isinstance(
        summary,
        DashboardSummary,
    )

    assert summary.latest_price == pytest.approx(
        1750.0
    )

    assert summary.projected_value == pytest.approx(
        1750.0 * 1.05**2
    )

    assert isinstance(
        summary.latest_date,
        pd.Timestamp,
    )


def test_build_dashboard_summary_uses_latest_real_price(
    dashboard_data: pd.DataFrame,
) -> None:
    summary = build_dashboard_summary(
        dashboard_data,
        DashboardConfig(),
    )

    assert (
        summary.inflation_adjusted_price
        > 0
    )


def test_build_dashboard_summary_includes_risk_metrics(
    dashboard_data: pd.DataFrame,
) -> None:
    summary = build_dashboard_summary(
        dashboard_data,
        DashboardConfig(),
    )

    assert summary.annualized_volatility >= 0

    assert summary.maximum_drawdown <= 0


def test_format_dashboard_summary_values_returns_all_metrics(
    dashboard_data: pd.DataFrame,
) -> None:
    summary = build_dashboard_summary(
        dashboard_data,
        DashboardConfig(),
    )

    formatted = (
        format_dashboard_summary_values(
            summary
        )
    )

    assert set(
        formatted
    ) == set(
        DASHBOARD_SUMMARY_ORDER
    )


def test_populate_summary_panel_adds_metric_labels(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    config = DashboardConfig()

    summary = build_dashboard_summary(
        dashboard_data,
        config,
    )

    result = populate_summary_panel(
        axes,
        summary,
        config,
    )

    assert result is axes

    text_values = [
        text.get_text()
        for text in axes.texts
    ]

    for metric_name in (
        DASHBOARD_SUMMARY_ORDER
    ):
        assert (
            DASHBOARD_SUMMARY_LABELS[
                metric_name
            ]
            in text_values
        )


def test_populate_summary_panel_adds_forecast_note(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    config = DashboardConfig(
        forecast_years=3,
        annual_growth_rate=0.06,
    )

    summary = build_dashboard_summary(
        dashboard_data,
        config,
    )

    populate_summary_panel(
        axes,
        summary,
        config,
    )

    text_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert any(
        "6.0% annual growth"
        in text
        for text in text_values
    )

    assert any(
        "3 years"
        in text
        for text in text_values
    )


# ------------------------------------------------------------------
# Price and inflation dashboard-panel tests
# ------------------------------------------------------------------


def test_format_dashboard_date_axis_returns_axes() -> None:
    _, axes = plt.subplots()

    result = format_dashboard_date_axis(
        axes
    )

    assert result is axes


def test_populate_historical_panel_draws_price_series(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    result = populate_historical_panel(
        axes,
        dashboard_data,
        DashboardConfig(),
    )

    assert result is axes
    assert len(axes.lines) == 1

    assert (
        axes.lines[0].get_label()
        == DASHBOARD_HISTORICAL_LABEL
    )


def test_populate_historical_panel_formats_currency_axis(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_historical_panel(
        axes,
        dashboard_data,
        DashboardConfig(),
    )

    assert isinstance(
        axes.yaxis.get_major_formatter(),
        FuncFormatter,
    )


def test_populate_recent_panel_limits_dates(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    config = DashboardConfig(
        recent_years=1,
    )

    populate_recent_panel(
        axes,
        dashboard_data,
        config,
    )

    x_values = axes.lines[0].get_xdata()

    assert len(x_values) < len(
        dashboard_data
    )


def test_populate_recent_panel_adds_window_label(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    config = DashboardConfig(
        recent_years=2,
    )

    populate_recent_panel(
        axes,
        dashboard_data,
        config,
    )

    assert (
        axes.lines[0].get_label()
        == "Most Recent 2 Years"
    )


def test_populate_nominal_vs_real_panel_draws_two_series(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_nominal_vs_real_panel(
        axes,
        dashboard_data,
        DashboardConfig(),
    )

    assert len(
        axes.lines
    ) == 2

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert labels == [
        DASHBOARD_NOMINAL_LABEL,
        DASHBOARD_REAL_LABEL,
    ]


def test_populate_price_panels_populates_all_three(
    dashboard_data: pd.DataFrame,
) -> None:
    figure = plt.figure()

    grid = create_dashboard_grid(
        figure
    )

    axes = create_dashboard_axes(
        figure,
        grid,
    )

    result = populate_price_panels(
        axes,
        dashboard_data,
        DashboardConfig(),
    )

    assert result is axes

    assert axes[
        PANEL_HISTORICAL_PRICE
    ].lines

    assert axes[
        PANEL_RECENT_PRICE
    ].lines

    assert len(
        axes[
            PANEL_NOMINAL_VS_REAL
        ].lines
    ) == 2


def test_build_dashboard_layout_populates_price_panels(
    dashboard_data: pd.DataFrame,
) -> None:
    result = (
        dashboard_module.build_dashboard_layout(
            dashboard_data
        )
    )

    assert result.axes[
        dashboard_module.PANEL_HISTORICAL_PRICE
    ].lines

    assert result.axes[
        dashboard_module.PANEL_RECENT_PRICE
    ].lines

    assert len(
        result.axes[
            dashboard_module.PANEL_NOMINAL_VS_REAL
        ].lines
    ) == 2

# ------------------------------------------------------------------
# Rolling-analysis dashboard-panel tests
# ------------------------------------------------------------------


def test_populate_rolling_volatility_panel_draws_series(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    result = populate_rolling_volatility_panel(
        axes,
        mixed_dashboard_data,
        DashboardConfig(
            rolling_window=6,
        ),
    )

    assert result is axes

    assert len(
        axes.lines
    ) == 1

    assert (
        DASHBOARD_VOLATILITY_LABEL
        in axes.lines[0].get_label()
    )


def test_populate_rolling_volatility_panel_uses_semantic_color(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_rolling_volatility_panel(
        axes,
        mixed_dashboard_data,
        DashboardConfig(
            rolling_window=6,
        ),
    )

    actual_color = (
        matplotlib.colors.to_rgba(
            axes.lines[0].get_color()
        )
    )

    expected_color = (
        matplotlib.colors.to_rgba(
            VOLATILITY_COLOR
        )
    )

    assert actual_color == expected_color


def test_populate_rolling_volatility_panel_uses_percentage_axis(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_rolling_volatility_panel(
        axes,
        mixed_dashboard_data,
        DashboardConfig(
            rolling_window=6,
        ),
    )

    assert isinstance(
        axes.yaxis.get_major_formatter(),
        FuncFormatter,
    )


def test_populate_rolling_drawdown_panel_draws_series_and_reference_line(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    result = populate_rolling_drawdown_panel(
        axes,
        mixed_dashboard_data,
        DashboardConfig(
            rolling_window=6,
        ),
    )

    assert result is axes

    assert len(
        axes.lines
    ) == 2

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert any(
        DASHBOARD_DRAWDOWN_LABEL
        in label
        for label in labels
    )

    assert (
        DASHBOARD_ROLLING_HIGH_LABEL
        in labels
    )


def test_populate_rolling_drawdown_panel_uses_fill(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_rolling_drawdown_panel(
        axes,
        mixed_dashboard_data,
        DashboardConfig(
            rolling_window=6,
        ),
    )

    assert axes.collections


def test_populate_rolling_drawdown_panel_uses_semantic_color(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_rolling_drawdown_panel(
        axes,
        mixed_dashboard_data,
        DashboardConfig(
            rolling_window=6,
        ),
    )

    actual_color = (
        matplotlib.colors.to_rgba(
            axes.lines[0].get_color()
        )
    )

    expected_color = (
        matplotlib.colors.to_rgba(
            DRAWDOWN_COLOR
        )
    )

    assert actual_color == expected_color


def test_populate_rolling_drawdown_panel_uses_percentage_axis(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_rolling_drawdown_panel(
        axes,
        mixed_dashboard_data,
        DashboardConfig(
            rolling_window=6,
        ),
    )

    assert isinstance(
        axes.yaxis.get_major_formatter(),
        FuncFormatter,
    )


def test_populate_rolling_panels_populates_both_panels(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    figure = plt.figure()

    grid = create_dashboard_grid(
        figure
    )

    axes = create_dashboard_axes(
        figure,
        grid,
    )

    result = populate_rolling_panels(
        axes,
        mixed_dashboard_data,
        DashboardConfig(
            rolling_window=6,
        ),
    )

    assert result is axes

    assert axes[
        PANEL_ROLLING_VOLATILITY
    ].lines

    assert axes[
        PANEL_ROLLING_DRAWDOWN
    ].lines


def test_populate_rolling_panels_rejects_missing_panels(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    with pytest.raises(
        ValueError,
        match=(
            "Dashboard axes are missing "
            "rolling panels"
        ),
    ):
        populate_rolling_panels(
            {
                PANEL_ROLLING_VOLATILITY: axes,
            },
            mixed_dashboard_data,
            DashboardConfig(
                rolling_window=6,
            ),
        )


def test_build_dashboard_layout_populates_rolling_panels(
    mixed_dashboard_data: pd.DataFrame,
) -> None:
    result = build_dashboard_layout(
        mixed_dashboard_data,
        config=DashboardConfig(
            rolling_window=6,
        ),
    )

    assert result.axes[
        PANEL_ROLLING_VOLATILITY
    ].lines

    assert result.axes[
        PANEL_ROLLING_DRAWDOWN
    ].lines

# ------------------------------------------------------------------
# Forecast dashboard-panel tests
# ------------------------------------------------------------------


def test_populate_forecast_panel_draws_expected_lines(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    result = populate_forecast_panel(
        axes,
        dashboard_data,
        DashboardConfig(
            forecast_years=2,
            forecast_history_years=2,
        ),
    )

    assert result is axes

    assert len(
        axes.lines
    ) == 3

    labels = [
        line.get_label()
        for line in axes.lines
    ]

    assert (
        DASHBOARD_FORECAST_HISTORY_LABEL
        in labels
    )

    assert any(
        DASHBOARD_FORECAST_LABEL
        in label
        for label in labels
    )


def test_populate_forecast_panel_connects_to_history(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_forecast_panel(
        axes,
        dashboard_data,
        DashboardConfig(
            forecast_years=1,
            forecast_history_years=2,
        ),
    )

    historical_line = axes.lines[0]
    forecast_line = axes.lines[1]

    assert (
        forecast_line.get_ydata()[0]
        == pytest.approx(
            historical_line.get_ydata()[-1]
        )
    )


def test_populate_forecast_panel_uses_semantic_colors(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_forecast_panel(
        axes,
        dashboard_data,
        DashboardConfig(
            forecast_years=1,
        ),
    )

    actual_colors = [
        matplotlib.colors.to_rgba(
            line.get_color()
        )
        for line in axes.lines[:3]
    ]

    expected_colors = [
        matplotlib.colors.to_rgba(
            HISTORICAL_PRICE_COLOR
        ),
        matplotlib.colors.to_rgba(
            FORECAST_COLOR
        ),
        matplotlib.colors.to_rgba(
            FORECAST_BOUNDARY_COLOR
        ),
    ]

    assert actual_colors == expected_colors


def test_populate_forecast_panel_labels_boundary(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_forecast_panel(
        axes,
        dashboard_data,
        DashboardConfig(),
    )

    text_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert (
        DASHBOARD_FORECAST_BOUNDARY_LABEL
        in text_values
    )


def test_populate_forecast_panel_adds_projected_value_annotation(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_forecast_panel(
        axes,
        dashboard_data,
        DashboardConfig(
            forecast_years=3,
            show_annotations=True,
        ),
    )

    text_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert any(
        text.startswith(
            "Projected 3-Year Value:"
        )
        for text in text_values
    )


def test_populate_forecast_panel_can_hide_projected_value(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_forecast_panel(
        axes,
        dashboard_data,
        DashboardConfig(
            show_annotations=False,
        ),
    )

    text_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert not any(
        text.startswith("Projected")
        and "Value:" in text
        for text in text_values
    )


def test_populate_forecast_panel_adds_disclaimer(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_forecast_panel(
        axes,
        dashboard_data,
        DashboardConfig(
            show_forecast_disclaimer=True,
        ),
    )

    text_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert (
        DASHBOARD_FORECAST_DISCLAIMER
        in text_values
    )


def test_populate_forecast_panel_can_hide_disclaimer(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_forecast_panel(
        axes,
        dashboard_data,
        DashboardConfig(
            show_forecast_disclaimer=False,
        ),
    )

    text_values = [
        text.get_text()
        for text in axes.texts
    ]

    assert (
        DASHBOARD_FORECAST_DISCLAIMER
        not in text_values
    )


def test_populate_forecast_panel_uses_currency_axis(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    populate_forecast_panel(
        axes,
        dashboard_data,
        DashboardConfig(),
    )

    assert isinstance(
        axes.yaxis.get_major_formatter(),
        FuncFormatter,
    )


def test_populate_forecast_panels_populates_panel(
    dashboard_data: pd.DataFrame,
) -> None:
    figure = plt.figure()

    grid = create_dashboard_grid(
        figure
    )

    axes = create_dashboard_axes(
        figure,
        grid,
    )

    result = populate_forecast_panels(
        axes,
        dashboard_data,
        DashboardConfig(),
    )

    assert result is axes

    assert axes[
        PANEL_FORECAST
    ].lines


def test_populate_forecast_panels_rejects_missing_panel(
    dashboard_data: pd.DataFrame,
) -> None:
    _, axes = plt.subplots()

    with pytest.raises(
        ValueError,
        match=(
            "Dashboard axes are missing "
            "the forecast panel"
        ),
    ):
        populate_forecast_panels(
            {
                PANEL_SUMMARY: axes,
            },
            dashboard_data,
            DashboardConfig(),
        )


def test_build_dashboard_layout_populates_forecast_panel(
    dashboard_data: pd.DataFrame,
) -> None:
    result = build_dashboard_layout(
        dashboard_data,
        config=DashboardConfig(
            forecast_years=2,
        ),
    )

    forecast_axes = result.axes[
        PANEL_FORECAST
    ]

    assert len(
        forecast_axes.lines
    ) == 3

    assert any(
        DASHBOARD_FORECAST_BOUNDARY_LABEL
        == text.get_text()
        for text in forecast_axes.texts
    )